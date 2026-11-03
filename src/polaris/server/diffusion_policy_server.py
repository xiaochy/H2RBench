"""
Policy inference server for DiffusionUnetHybridImagePolicy.
Run this in the robodiff conda env BEFORE running eval in the polaris uv env.

Usage:
    conda activate robodiff
    cd ~/vxiao/human2robot/benchmark_new/polaris
    python src/polaris/server/diffusion_policy_server.py \
        --ckpt_path <path/to/checkpoint.ckpt> \
        --port 5556
"""

import argparse
import pickle
import sys
import pathlib

import numpy as np
import torch
import zmq
import dill
from collections import deque

# Make sure diffusion_policy is importable
ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "policy" / "diffusion_policy")
sys.path.insert(0, ROOT_DIR)

# Stub out wandb before importing the workspace — the server doesn't need it
# and wandb tries to create tempfiles at import time which can fail
from unittest.mock import MagicMock
import types
_wandb_mock = MagicMock()
_wandb_mock.__spec__ = types.ModuleType('wandb')
sys.modules['wandb'] = _wandb_mock

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace


def load_policy(ckpt_path: str):
    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.load_payload(payload)
    if cfg.training.use_ema:
        policy = workspace.ema_model
    else:
        policy = workspace.model
    policy.eval()
    policy.to("cuda:0")
    return policy, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()

    print(f"Loading policy from {args.ckpt_path}...")
    policy, cfg = load_policy(args.ckpt_path)
    n_obs_steps = cfg.n_obs_steps  # 2
    print(f"Policy loaded. n_obs_steps={n_obs_steps}")

    image_buf    = deque(maxlen=n_obs_steps)
    agent_pos_buf = deque(maxlen=n_obs_steps)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Server listening on port {args.port}")

    while True:
        raw = socket.recv()
        try:
            request = pickle.loads(raw)
        except Exception as e:
            print(f"[Server] Failed to deserialize request: {e}, skipping.")
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))
            continue

        if request.get("reset"):
            image_buf.clear()
            agent_pos_buf.clear()
            socket.send(pickle.dumps({"status": "ok"}))
            continue

        # image: (H, W, 3) uint8  →  (3, H, W) float in [0,1]
        image = request["image"]  # already resized to (240, 426, 3)
        agent_pos = request["agent_pos"]  # (10,) float32

        img_tensor = torch.from_numpy(image).float() / 255.0  # (240, 426, 3)
        img_tensor = img_tensor.permute(2, 0, 1)              # (3, 240, 426)

        image_buf.append(img_tensor)
        agent_pos_buf.append(torch.from_numpy(agent_pos).float())

        # Pad buffer if not full yet (repeat first obs)
        while len(image_buf) < n_obs_steps:
            image_buf.appendleft(image_buf[0])
            agent_pos_buf.appendleft(agent_pos_buf[0])

        # Stack to (1, T, ...)
        images    = torch.stack(list(image_buf), dim=0).unsqueeze(0).to("cuda:0")     # (1, T, 3, 240, 426)
        agent_pos_t = torch.stack(list(agent_pos_buf), dim=0).unsqueeze(0).to("cuda:0")  # (1, T, 10)

        obs_dict = {
            "image":     images,
            "agent_pos": agent_pos_t,
        }

        with torch.no_grad():
            result = policy.predict_action(obs_dict)

        # action: (1, n_action_steps, 10)  →  (n_action_steps, 10)
        action_chunk = result["action"].squeeze(0).cpu().numpy()

        socket.send(pickle.dumps({"action_chunk": action_chunk}))


if __name__ == "__main__":
    main()
