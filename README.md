<p align="center">
  <img src="assets/figures/logo.png" alt="VLX-Go logo" width="180">
</p>

<h1 align="center">VLX-Go</h1>

<h3 align="center">Vision-Language Short-Horizon Waypoint Prediction for Embodied Navigation</h3>

<p align="center">
  English | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://x.com/OmAI_lab"><img src="https://img.shields.io/badge/%F0%9F%93%A3%20X-Follow%20%40OmAI_lab-000000" alt="Follow @OmAI lab on X"></a>
  <a href="TBD"><img src="https://img.shields.io/badge/%F0%9F%93%9D%20Blog-Read%20Article-2563eb" alt="Blog article"></a>
  <a href="#model-checkpoint"><img src="https://img.shields.io/badge/Model-Coming%20Soon-111827" alt="Model coming soon"></a>
  <a href="#dataset"><img src="https://img.shields.io/badge/Dataset-Coming%20Soon-111827" alt="Dataset coming soon"></a>
  <a href="TBD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Read%20Blog-f9d54a" alt="Hugging Face blog"></a>
</p>

<p align="center"><sub>Overview video: VLX-Go for vision-language short-horizon waypoint prediction in embodied navigation.</sub></p>

https://github.com/user-attachments/assets/112a5a86-847b-4968-8469-c022bb20f75d

## Overview

<p align="center">
  <img src="assets/figures/overall_intro.png" alt="VLX-Go overview: video history, current visual state, and language instruction are converted into short-horizon waypoints" width="88%">
</p>

VLX-Go is a lightweight vision-language waypoint planner for embodied navigation. Given recent monocular frames, the current observation, and a natural-language instruction, it predicts short-horizon local waypoints that can be executed by a downstream controller in simulation or on a robot.

Rather than relying on a general-purpose VLM to describe the scene or produce text-only actions, VLX-Go maps the visual-language state directly to a compact waypoint interface. This repository focuses on target following, local navigation, dynamic obstacle avoidance, and closed-loop evaluation.

VLX-Go builds on the technical direction of [OmTrackVLA](https://github.com/om-ai-lab/OmTrackVLA) and extends it toward lightweight waypoint prediction and closed-loop navigation research.

## Highlights

- **0.6B waypoint planner:** A lightweight architecture for lower inference cost and easier deployment.
- **Instruction-conditioned planning:** Natural-language instructions specify task intent, including following, reaching, and obstacle avoidance.
- **Short-horizon waypoint output:** The model predicts local motion targets instead of full global routes or text-only responses.
- **Temporal visual context:** Recent frames help capture target motion, occlusion changes, and scene dynamics.
- **Closed-loop evaluation:** Navigation is evaluated through repeated observation, prediction, execution, and feedback.

## Task

At each time step, VLX-Go solves a receding-horizon waypoint prediction problem.

**Input**

- Recent visual history: `H_t = {I_{t-k}, ..., I_{t-1}}`
- Current frame: `I_t`
- Instruction: `q`, for example, "follow the target person and avoid obstacles"

**Output**

- Short-horizon waypoint sequence: `W_t = {w_1, ..., w_T}`

Here, each `w_i` denotes a local motion target, such as position, heading, or another waypoint representation consumed by the controller. The exact dimensionality depends on the dataset and control interface.

```text
history frames + current frame + instruction
                 |
                 v
        VLX-Go waypoint planner
                 |
                 v
       short-horizon waypoints -> controller / simulator
```

## Method

VLX-Go separates waypoint planning from platform-specific low-level control. The planner predicts short-horizon local goals, while the downstream controller handles velocity commands, safety constraints, and dynamics.

| Stage | Description |
| --- | --- |
| Visual encoding | Encode the current frame and recent visual history into visual features |
| Language conditioning | Use the instruction as the task condition for planning |
| Waypoint prediction | Predict short-horizon local motion targets with a 0.6B planner |
| Closed-loop execution | Execute predicted waypoints, collect the next observation, and predict the next segment |

This rolling-horizon design is practical for dynamic scenes: targets may move, obstacles may enter the camera view, and earlier predictions can be corrected with new observations.

## Training

VLX-Go is first trained from offline trajectory data and can then be refined with online simulator feedback.

| Phase | Data / Signal | Objective |
| --- | --- | --- |
| Offline trajectory learning | Demonstration trajectories, video frames, language instructions | Learn target-following and local waypoint generation |
| Online optimization | Simulator feedback, collision signals, target state, reward signals | Improve robustness to occlusion, obstacles, and closed-loop drift |

Typical supervised objectives include waypoint regression, trajectory direction loss, optional velocity or action auxiliary loss, and smoothness regularization. The online stage complements supervised learning by exposing the policy to execution-time feedback that offline trajectory data may not cover.

## Evaluation

VLX-Go is evaluated on the STT task of EVT-Bench.

| Model | Parameters | STT SR ↑ | STT TR ↑ | STT CR ↓ |
| --- | --- | ---: | ---: | ---: |
| TrackVLA | 7B | 85.1 | 78.6 | **1.65** |
| NavFoM | 7B | 85.0 | 80.5 | - |
| Qwen-RobotNav-4B | 4B | 77.4 | 90.0 | 6.4 |
| Qwen-RobotNav-8B | 8B | 78.6 | 89.7 | 5.7 |
| **VLX-Go** | **0.6B** | **85.42** | **94.08** | 6.55 |

**Metrics:** SR is success rate, TR is tracking rate, and CR is collision rate. At the 0.6B scale, VLX-Go achieves strong success and tracking rates. Further reducing collision rate remains an important direction, especially through simulator, reward, controller, and safety-constraint tuning.

## Model Checkpoint

Coming soon.

## Dataset

Coming soon.

## Follow us

Follow Om AI Lab on [X](https://x.com/OmAI_lab), or scan the WeChat group QR code below for VLX updates and discussion.

<p align="left">
  <img src="assets/figures/wechat_qr.png" alt="WeChat community QR code" width="200">
</p>
