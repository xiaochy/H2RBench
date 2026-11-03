"""
Policy inference server for LeRobot Diffusion policy.
Run this in the robodiff conda env BEFORE running eval.py in the polaris uv env.

Usage:
    conda activate robodiff
    python scripts/lerobot_diffusion_server.py --policy_path <path> --open_loop_horizon <n>
"""

import argparse
import pickle
import numpy as np
import torch
from collections import deque

import zmq

from lerobot.common.policies.pretrained import PreTrainedConfig
from lerobot.common.policies.factory import get_policy_class
import lerobot
import os


N_OBS_STEPS = 2


def load_policy(policy_path: str):
    policy_config: PreTrainedConfig = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path

    lerobot_dir = os.path.dirname(os.path.dirname(lerobot.__file__))
    policy_config.calibration_json = os.path.join(lerobot_dir, policy_config.calibration_json)

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(config=policy_config, pretrained_name_or_path=policy_path)
    policy.eval()
    policy.to("cuda")
    return policy


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0  # (H, W, 3)
    t = t.permute(2, 0, 1)                     # (3, H, W)
    return t.unsqueeze(0).to("cuda")           # (1, 3, H, W)


def build_lerobot_obs(exterior_buf, wrist_buf, state_buf, instruction: str) -> dict:
    state_tensor = torch.from_numpy(state_buf[-1]).float().unsqueeze(0).to("cuda")
    return {
        "observation.images.cam_azure_kinect_front.color": img_to_tensor(exterior_buf[-1]),
        "observation.images.cam_wrist":                    img_to_tensor(wrist_buf[-1]),
        "observation.state":                               state_tensor,
        "task":                                            [instruction],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--open_loop_horizon", type=int, required=True)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    print(f"Loading policy from {args.policy_path}...")
    policy = load_policy(args.policy_path)
    print("Policy loaded.")

    exterior_buf = deque(maxlen=N_OBS_STEPS)
    wrist_buf    = deque(maxlen=N_OBS_STEPS)
    state_buf    = deque(maxlen=N_OBS_STEPS)
    actions_from_chunk_completed = 0
    open_loop_horizon = args.open_loop_horizon

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Server listening on port {args.port}")

    while True:
        raw = socket.recv()
        request = pickle.loads(raw)

        if request.get("reset"):
            exterior_buf.clear()
            wrist_buf.clear()
            state_buf.clear()
            actions_from_chunk_completed = 0
            policy.reset()
            socket.send(pickle.dumps({"status": "ok"}))
            continue

        cam1      = request["cam1"]       # (H, W, 3) uint8
        wrist_cam = request["wrist_cam"]  # (H, W, 3) uint8
        state     = request["state"]      # (8,) float32
        instruction = request["instruction"]

        exterior_buf.append(cam1)
        wrist_buf.append(wrist_cam)
        state_buf.append(state)

        lerobot_obs = build_lerobot_obs(exterior_buf, wrist_buf, state_buf, instruction)

        with torch.inference_mode():
            action_tensor, _ = policy.select_action(lerobot_obs)

        action = action_tensor.squeeze(0).cpu().numpy()

        # Binarize gripper
        action = np.concatenate([
            action[:-1],
            np.ones((1,)) if action[-1] > 0.5 else np.zeros((1,))
        ])

        actions_from_chunk_completed += 1
        if actions_from_chunk_completed >= open_loop_horizon:
            actions_from_chunk_completed = 0

        rerender = (actions_from_chunk_completed == 0)
        socket.send(pickle.dumps({"action": action, "rerender": rerender}))


if __name__ == "__main__":
    main()
