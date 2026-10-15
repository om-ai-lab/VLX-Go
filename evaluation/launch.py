"""Launch EVT-Bench from eval.sh and manage only workers created by this invocation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .constants import REPO_ROOT, TASKS


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="VLX-Go EVT-Bench evaluation (HF checkpoints only)")
    parser.add_argument("--task", nargs="+", default=[os.environ.get("TASKS", "dt stt at")],
                        help="stt, dt, at, all, or a quoted task list")
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS") or None)
    parser.add_argument("--num-parallel", default=os.environ.get("NUM_PARALLEL", "1"))
    parser.add_argument("--jobs-per-gpu", type=positive_int,
                        default=os.environ.get("JOBS_PER_GPU", "1"))
    parser.add_argument("--no-save-video", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    parser.add_argument("--max-episodes", type=positive_int,
                        help="run a bounded subset per task in one worker instead of all 201 splits")
    parser.add_argument("--check-assets", action="store_true",
                        help="check environment/configuration and asset paths without loading models or a simulator")
    args = parser.parse_args()
    tasks = " ".join(args.task).split()
    if tasks == ["all"]:
        tasks = list(TASKS)
    if not tasks or any(task not in TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        parser.error("tasks must be distinct stt, dt, at, or all")
    if args.gpu_ids is None:
        try:
            gpus = list(range(positive_int(args.num_parallel)))
        except argparse.ArgumentTypeError as exc:
            parser.error(f"--num-parallel / NUM_PARALLEL: {exc}")
    else:
        if not re.fullmatch(r"\s*[0-9]+(?:\s*,\s*[0-9]+|\s+[0-9]+)*\s*", args.gpu_ids):
            parser.error("--gpu-ids / GPU_IDS must contain non-negative IDs separated by commas or spaces")
        gpus = [int(item) for item in re.split(r"[\s,]+", args.gpu_ids.strip())]
        if len(set(gpus)) != len(gpus):
            parser.error("duplicate GPU IDs")
    for name in ("RESUME", "SAVE_VIDEO", "SAVE_THIRD_PERSON"):
        if os.environ.get(name, "1") not in {"0", "1"}:
            parser.error(f"{name} must be 0 or 1")
    do_resume = args.resume if args.resume is not None else os.environ.get("RESUME", "1") == "1"
    if os.environ.get("CKPT"):
        parser.error("CKPT is not supported; unset CKPT and select a complete HF export with HF_MODEL_DIR")

    # Resolve the caller's relative paths before switching to the repository directory.
    env = os.environ.copy()
    for name in ("HF_MODEL_DIR", "DINOV3_MODEL_PATH", "SIGLIP_MODEL_PATH"):
        value = env.get(name)
        if value and (Path(value).expanduser().exists() or value.startswith(("/", ".", "~"))):
            env[name] = str(Path(value).expanduser().resolve())
    output = Path(os.environ.get("EVAL_OUT_DIR", REPO_ROOT / "results")).expanduser().resolve()
    suffix = os.environ.get("SAVE_SUFFIX", "")
    if any(char in suffix for char in ("/", "\\")) or suffix in {".", ".."}:
        parser.error("SAVE_SUFFIX must be a directory-name suffix, not a path")
    env.setdefault("HF_LOCAL_FILES_ONLY", "1")
    env.setdefault("VLXGO_ATTN_IMPL", "sdpa")
    env["PYTHONUNBUFFERED"] = "1"
    if args.no_save_video:
        env["SAVE_VIDEO"] = env["SAVE_THIRD_PERSON"] = "0"
    # Do not distribute local EGL symlinks; select the NVIDIA ICD only when it is installed.
    nvidia_icd = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if nvidia_icd.is_file() and not env.get("__EGL_VENDOR_LIBRARY_DIRS"):
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(nvidia_icd))

    split_count = 1 if args.max_episodes else 200
    split_ids = [0] if args.max_episodes else list(range(201))

    def command(task, split_id, *, preflight=False):
        task_output = output / (f"{task}_{suffix}" if suffix else task)
        argv = [sys.executable, "-m", "evaluation.run_eval", "--task", task,
                "--split-num", str(split_count), "--split-id", str(split_id),
                "--save-path", str(task_output), "--resume" if do_resume else "--no-resume"]
        if args.max_episodes:
            argv += ["--max-episodes", str(args.max_episodes)]
        if preflight:
            argv += ["--check-assets"]
        return argv

    # Check assets for all selected tasks before launching workers, reporting missing assets once.
    for task in tasks:
        rc = subprocess.call(command(task, 0, preflight=True), cwd=REPO_ROOT, env=env)
        if rc:
            return rc
    if args.check_assets:
        return 0

    print(f"[eval] tasks={tasks}; GPUs={gpus}; jobs/GPU={args.jobs_per_gpu}; "
          f"splits/task={len(split_ids)}; resume={do_resume}; output={output}", flush=True)
    active = {}
    failures = 0

    def stop_workers(signum, _frame):
        # Terminate only process groups created here, without matching or cleaning up other evaluation processes.
        for process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_workers)
    signal.signal(signal.SIGINT, stop_workers)
    try:
        for task in tasks:
            task_output = output / (f"{task}_{suffix}" if suffix else task)
            log_dir = task_output / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            pending = list(split_ids)
            while pending or active:
                for split_id in pending[:]:
                    gpu = gpus[split_id % len(gpus)]
                    if sum(value[1] == gpu for value in active.values()) >= args.jobs_per_gpu:
                        continue
                    log_path = log_dir / f"split_{split_id:03d}.log"
                    stream = log_path.open("a", buffering=1)
                    stream.write(f"\n[launcher] {time.strftime('%Y-%m-%d %H:%M:%S')} GPU={gpu}\n")
                    child_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    try:
                        process = subprocess.Popen(command(task, split_id), cwd=REPO_ROOT,
                                                   env=child_env, stdout=stream,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    finally:
                        stream.close()
                    active[split_id] = (process, gpu, log_path)
                    pending.remove(split_id)
                    print(f"[eval] {task} split={split_id} GPU={gpu} log={log_path}", flush=True)
                for split_id, (process, gpu, log_path) in list(active.items()):
                    rc = process.poll()
                    if rc is not None:
                        failures += int(rc != 0)
                        del active[split_id]
                        print(f"[eval] {task} split={split_id} exit={rc}", flush=True)
                if active:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        print("[eval] Interrupted; unfinished episodes can be resumed.", file=sys.stderr)
        return 130
    finally:
        for process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for process, _, _ in active.values():
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    print(f"[eval] Finished; failed splits={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
