from typing import Dict, List, Optional, Set

import attr
import magnum as mn
import numpy as np

from habitat.articulated_agents.articulated_agent_base import (
    ArticulatedAgentBase,
)
from habitat.articulated_agents.manipulator import Manipulator
from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
)


@attr.s(auto_attribs=True, slots=True)
class Go2Params:
    """Parameter bundle for a base-only Unitree Go2 robot."""

    arm_joints: List[int]
    gripper_joints: List[int]

    arm_init_params: Optional[np.ndarray]
    gripper_init_params: Optional[np.ndarray]

    ee_offset: List[mn.Vector3]
    ee_links: List[int]
    ee_constraint: np.ndarray

    cameras: Dict[str, ArticulatedAgentCameraParams]

    gripper_closed_state: np.ndarray
    gripper_open_state: np.ndarray
    gripper_state_eps: float

    arm_mtr_pos_gain: float
    arm_mtr_vel_gain: float
    arm_mtr_max_impulse: float

    base_offset: mn.Vector3
    base_link_names: Set[str]

    leg_joints: Optional[List[int]] = None
    leg_init_params: Optional[np.ndarray] = None
    leg_mtr_pos_gain: Optional[float] = None
    leg_mtr_vel_gain: Optional[float] = None
    leg_mtr_max_impulse: Optional[float] = None

    ee_count: Optional[int] = 0


class Go2Robot(MobileManipulator):
    """Minimal Go2 wrapper for tracking tasks without a manipulator arm."""

    LEG_JOINT_NAMES = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    def _get_go2_params(self) -> Go2Params:
        standing_pose = np.array([0.0, 0.7, -1.5] * 4, dtype=np.float32)
        camera_pos = mn.Vector3(0.25, 0.15, 0.0)
        front_camera_yaw = -np.pi / 2
        return Go2Params(
            arm_joints=[],
            gripper_joints=[],
            arm_init_params=np.array([], dtype=np.float32),
            gripper_init_params=np.array([], dtype=np.float32),
            ee_offset=[],
            ee_links=[],
            ee_constraint=np.zeros((0, 2, 3), dtype=np.float32),
            cameras={
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.55, 1.35, 0.0),
                    cam_look_at_pos=mn.Vector3(0.8, 0.15, 0.0),
                    attached_link_id=-1,
                ),
                "articulated_agent_jaw_depth": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.30, 0.12, 0.0),
                    cam_orientation=mn.Vector3(0.0, -1.571, 0.0),
                    attached_link_id=-1,
                ),
                "articulated_agent_jaw_rgb": ArticulatedAgentCameraParams(
                    cam_offset_pos=camera_pos,
                    cam_orientation=mn.Vector3(0.0, front_camera_yaw, 0.0),
                    attached_link_id=-1,
                ),
                "articulated_agent_right_rgb": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 0.15, 0.15),
                    cam_orientation=mn.Vector3(0.0, front_camera_yaw - (np.pi / 2), 0.0),
                    attached_link_id=-1,
                ),
                "articulated_agent_back_rgb": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.25, 0.15, 0.0),
                    cam_orientation=mn.Vector3(0.0, front_camera_yaw + np.pi, 0.0),
                    attached_link_id=-1,
                ),
                "articulated_agent_left_rgb": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 0.15, -0.15),
                    cam_orientation=mn.Vector3(0.0, front_camera_yaw + (np.pi / 2), 0.0),
                    attached_link_id=-1,
                ),
                "articulated_agent_jaw_panoptic": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.25, 0.15, 0.0),
                    cam_orientation=mn.Vector3(0.0, front_camera_yaw, 0.0),
                    attached_link_id=-1,
                ),
            },
            gripper_closed_state=np.array([], dtype=np.float32),
            gripper_open_state=np.array([], dtype=np.float32),
            gripper_state_eps=0.0,
            arm_mtr_pos_gain=0.0,
            arm_mtr_vel_gain=0.0,
            arm_mtr_max_impulse=0.0,
            base_offset=mn.Vector3(0.0, -0.32, 0.0),
            base_link_names={"base_root", "base", "Head_upper", "Head_lower"},
            leg_joints=None,
            leg_init_params=standing_pose,
            leg_mtr_pos_gain=2.0,
            leg_mtr_vel_gain=1.3,
            leg_mtr_max_impulse=80.0,
        )

    def __init__(
        self, agent_cfg, sim, limit_robo_joints=True, fixed_base=True
    ):
        self._virtual_gripper_closed = False
        super().__init__(
            self._get_go2_params(),
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
            maintain_link_order=True,
            base_type="leg",
        )

    def _resolve_leg_joint_ids(self) -> List[int]:
        joint_name_to_link_id = {}
        for link_id in self.sim_obj.get_link_ids():
            joint_name = self.sim_obj.get_link_joint_name(link_id)
            if joint_name:
                joint_name_to_link_id[joint_name] = link_id

        missing_joint_names = [
            joint_name
            for joint_name in self.LEG_JOINT_NAMES
            if joint_name not in joint_name_to_link_id
        ]
        if missing_joint_names:
            available = self.get_link_and_joint_names()
            raise ValueError(
                "Failed to resolve Go2 leg joints. "
                f"Missing joints: {missing_joint_names}. "
                f"Available links/joints:\n{available}"
            )

        return [
            joint_name_to_link_id[joint_name]
            for joint_name in self.LEG_JOINT_NAMES
        ]

    def reconfigure(self) -> None:
        Manipulator.reconfigure(self)
        self.params.leg_joints = self._resolve_leg_joint_ids()
        ArticulatedAgentBase.reconfigure(self)
        self.update()

    def reset(self) -> None:
        self._virtual_gripper_closed = False
        super().reset()
        self.update()

    def update(self) -> None:
        super().update()

    @property
    def base_transformation(self):
        return self.sim_obj.transformation

    def open_gripper(self) -> None:
        self._virtual_gripper_closed = False

    def close_gripper(self) -> None:
        self._virtual_gripper_closed = True

    @property
    def is_gripper_open(self) -> bool:
        return not self._virtual_gripper_closed

    @property
    def is_gripper_closed(self) -> bool:
        return self._virtual_gripper_closed
