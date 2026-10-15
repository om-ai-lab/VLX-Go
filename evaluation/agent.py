"""Preserve EVT-Bench environment interactions and scoring timing while reusing vlxgo.runtime for inference."""
from __future__ import annotations

import os
import os.path as osp
import json
from typing import Optional

import habitat
import imageio.v2 as imageio
import numpy as np
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import trange

from vlxgo.attention import configure_torch_inference_backends
from vlxgo.runtime import EngineRuntime
from .resume import atomic_write_result, episode_result_path


def evaluate_agent(config, dataset_split, save_path) -> None:
    # Match the video frame rate to the simulation action frequency and retain the original control interface.
    robot_config = WaypointAgent(
        save_path,
        config.habitat.simulator.ctrl_freq / config.habitat.simulator.ac_freq_ratio,
    )
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            result_path = episode_result_path(save_path, env.current_episode)
            # Once re-evaluation starts, the old final result cannot mark this rollout as complete.
            result_path.unlink(missing_ok=True)
            light_setup = [
                LightInfo(
                    vector=[10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[-10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, 10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, -10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []

            # Use the annotated instruction; shared inference supplies the default when it is missing.
            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None

            action_dict = dict()
            finished = False
            
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            human_no_move = 0
            too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()

            while not env.episode_over:
                record_info = {}
                
                obs = sim.get_sensor_observations()

                detector = env.task._get_observations(env.current_episode)
                action = robot_config.act(obs, detector, env.current_episode.episode_id, instruction, scene_id=env.current_episode.scene_id)

                action_dict = {
                    "action": (
                        "agent_0_humanoid_navigate_action",
                        "agent_1_base_velocity",
                        "agent_2_oracle_nav_randcoord_action_obstacle",
                        "agent_3_oracle_nav_randcoord_action_obstacle",
                        "agent_4_oracle_nav_randcoord_action_obstacle",
                        "agent_5_oracle_nav_randcoord_action_obstacle",
                    ),
                    "action_args": {
                        "agent_1_base_vel": action
                    }
                }
                
                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    followed_step += 1
                    too_far_count = 0

                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = 'Lost'
                        finished = False
                        break

                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_infos.append(record_info)

                if info['human_collision'] == 1.0:
                    print("Collision detected!")
                    status = 'Collision'
                    finished = False
                    break
                

            print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()
            robot_config.reset(env.current_episode)

            if env.episode_over:
                finished = True
            
            save_dir = result_path.parent
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2) 
            result['finish'] = finished
            result['status'] = status
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / iter_step
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            if instruction is not None:
                result['instruction'] = instruction
            atomic_write_result(result_path, result)


class WaypointAgent:
    """Use one model per worker and one history per episode; propagate inference errors."""

    def __init__(self, result_path, fps=10):
        self.result_path = result_path
        self.fps = fps
        self.save_video = os.environ.get("SAVE_VIDEO", "1") == "1"
        self.save_third_person = os.environ.get("SAVE_THIRD_PERSON", "1") == "1"
        self._clear_video_buffers()
        configure_torch_inference_backends()
        self.runtime = EngineRuntime()
        self._last_predicted_traj = None

    def reset(self, episode=None):
        if episode is not None:
            self._save_episode_videos(episode)
        self.runtime.clear_cache("episode")
        self._clear_video_buffers()

    def act(self, observations, detector, episode_id, instruction=None, scene_id=None):
        rgb = observations["agent_1_articulated_agent_jaw_rgb"][..., :3]
        result = self.runtime.infer("episode", rgb, instruction)
        self._last_predicted_traj = np.asarray(result["waypoints"], dtype=np.float32)
        if self.save_video:
            self.rgb_list.append(self._render_frame_with_traj(rgb, self._last_predicted_traj))
            for key, buffer in (("left", self.left_rgb_list),
                                ("right", self.right_rgb_list), ("back", self.rear_rgb_list)):
                self._append_optional_view_frame(observations,
                    f"agent_1_articulated_agent_{key}_rgb", buffer)
        third = self._get_third_person_frame(observations)
        if third is not None and self.save_third_person:
            self.third_rgb_list.append(third)
        return result["action"]

    def _save_episode_videos(self, episode) -> None:
        if (
            not self.rgb_list
            and not self.left_rgb_list
            and not self.right_rgb_list
            and not self.rear_rgb_list
            and not self.third_rgb_list
        ):
            return

        scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
        save_dir = os.path.join(self.result_path, scene_key)
        os.makedirs(save_dir, exist_ok=True)

        if self.save_video and self.rgb_list:
            output_video_path = os.path.join(save_dir, f"{episode.episode_id}.mp4")
            imageio.mimsave(output_video_path, self.rgb_list, fps=self.fps)
            self._save_optional_view_video(save_dir, episode.episode_id, "left", self.left_rgb_list)
            self._save_optional_view_video(save_dir, episode.episode_id, "right", self.right_rgb_list)
            self._save_optional_view_video(save_dir, episode.episode_id, "rear", self.rear_rgb_list)
            self._save_all_view_video(save_dir, episode.episode_id)
            print(f"Successfully saved first-person video for episode {episode.episode_id}")

        if self.save_third_person and self.third_rgb_list:
            third_person_video_path = os.path.join(save_dir, f"{episode.episode_id}_third_person.mp4")
            imageio.mimsave(third_person_video_path, self.third_rgb_list, fps=self.fps)
            print(f"Successfully saved third-person video for episode {episode.episode_id}")

        self._clear_video_buffers()

    def _get_third_person_frame(self, observations) -> Optional[np.ndarray]:
        third_rgb = observations.get("agent_1_third_rgb")
        if third_rgb is None:
            return None

        frame = np.asarray(third_rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None

        return frame[:, :, :3].astype(np.uint8)

    def _get_optional_rgb_frame(self, observations, key: str) -> Optional[np.ndarray]:
        rgb = observations.get(key)
        if rgb is None:
            return None

        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None

        return frame[:, :, :3].astype(np.uint8)

    def _append_optional_view_frame(self, observations, key: str, buffer_list) -> None:
        frame = self._get_optional_rgb_frame(observations, key)
        if frame is not None:
            buffer_list.append(frame)

    def _save_optional_view_video(self, save_dir: str, episode_id, suffix: str, frames) -> None:
        if not frames:
            return

        output_video_path = os.path.join(save_dir, f"{episode_id}_{suffix}.mp4")
        imageio.mimsave(output_video_path, frames, fps=self.fps)

    def _save_all_view_video(self, save_dir: str, episode_id) -> None:
        all_frames = self._build_all_view_frames()
        if not all_frames:
            return

        output_video_path = os.path.join(save_dir, f"{episode_id}_all.mp4")
        imageio.mimsave(output_video_path, all_frames, fps=self.fps)

    def _build_all_view_frames(self) -> list[np.ndarray]:
        if not (
            self.left_rgb_list
            and self.rgb_list
            and self.right_rgb_list
            and self.rear_rgb_list
        ):
            return []

        num_frames = min(
            len(self.left_rgb_list),
            len(self.rgb_list),
            len(self.right_rgb_list),
            len(self.rear_rgb_list),
        )
        all_frames = []
        for idx in range(num_frames):
            view_frames = [
                self.left_rgb_list[idx],
                self.rgb_list[idx],
                self.right_rgb_list[idx],
                self.rear_rgb_list[idx],
            ]
            min_height = min(frame.shape[0] for frame in view_frames)
            stitched = np.concatenate(
                [frame[:min_height, :, :3] for frame in view_frames],
                axis=1,
            )
            all_frames.append(stitched.astype(np.uint8))

        return all_frames

    def _clear_video_buffers(self) -> None:
        self.rgb_list = []
        self.left_rgb_list = []
        self.right_rgb_list = []
        self.rear_rgb_list = []
        self.third_rgb_list = []

    def _render_frame_with_traj(self, rgb_frame_np: np.ndarray, traj_xyz: Optional[np.ndarray]) -> np.ndarray:
        """Overlay predicted trajectory onto the frame and return the composited RGB frame."""
        try:
            if traj_xyz is None or not isinstance(traj_xyz, np.ndarray) or traj_xyz.size == 0:
                return rgb_frame_np
            from PIL import Image, ImageDraw
            img = Image.fromarray(rgb_frame_np.astype(np.uint8), mode='RGB')
            draw = ImageDraw.Draw(img)
            w, h = img.size
            base_x = w // 2
            base_y = int(h * 0.86)
            scale = 120.0  # px per meter
            pts = []
            npts = min(int(traj_xyz.shape[0]), 64)
            for i in range(npts):
                x = float(traj_xyz[i, 0])
                y = float(traj_xyz[i, 1]) if traj_xyz.shape[1] >= 2 else 0.0
                px = base_x - int(y * scale)
                py = base_y - int(x * scale)
                pts.append((px, py))
            # outline thicker black for visibility
            for i in range(1, len(pts)):
                draw.line([pts[i-1], pts[i]], fill=(0, 0, 0), width=8)
            # main colored line
            for i in range(1, len(pts)):
                draw.line([pts[i-1], pts[i]], fill=(0, 255, 180), width=4)
            # start point marker
            if pts:
                r = 4
                sx, sy = pts[0]
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(0, 255, 0))
            return np.asarray(img)
        except Exception:
            return rgb_frame_np
