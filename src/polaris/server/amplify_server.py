"""
AMPLIFY policy inference server for pick_mug task.
Run in the 'amplify' conda env BEFORE running eval in the polaris uv env.

Usage:
    conda activate amplify
    cd /home/veraxiao/vxiao/human2robot/benchmark_new/AMPLIFY
    python ../polaris/scripts/amplify_server.py \\
        --mt_ckpt  ../polaris/policy_ckpt/amplify/motion_tokenizer/motion_human0_robot40/latest.pt \\
        --fd_ckpt  ../polaris/policy_ckpt/amplify/forward_dynamics/pick_mug_human0_robot40/latest.pt \\
        --id_ckpt  ../polaris/policy_ckpt/amplify/inverse_dynamics/id_pick_mug_human0_robot40_seed_0/latest.pt \\
        --text_emb ../polaris/policy_ckpt/amplify/pick_mug_text_emb.npy \\
        --action_stats ../polaris/policy_ckpt/amplify/pick_mug_action_stats.npy \\
        --port 5557

Or with a pre-bundled AMPLIFY checkpoint:
    python ../polaris/scripts/amplify_server.py \\
        --ckpt_path ../polaris/policy_ckpt/amplify/bundled/pick_mug_human0_robot40.pt \\
        --text_emb  ../polaris/policy_ckpt/amplify/pick_mug_text_emb.npy \\
        --action_stats ../polaris/policy_ckpt/amplify/pick_mug_action_stats.npy \\
        --port 5557

Request format  (pickle):
    {"image": np.ndarray (v, 240, 426, 3) float32 [0,1],
     "proprio": np.ndarray (8,) float32}
    or {"reset": True}

Response format (pickle):
    {"action_chunk": np.ndarray (action_horizon, 8) float32}
    actions are in denormalized joint-position space.
"""

import argparse
import pickle
import sys
import pathlib

import numpy as np
import torch
import zmq
from einops import repeat

# Parse --amplify_root early so imports work before argparse runs fully
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--amplify_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()

ROOT_DIR = _pre_args.amplify_root or str(pathlib.Path(__file__).parent.parent / "policy" / "amplify")
POLARIS_SRC = str(pathlib.Path(__file__).parent.parent.parent)  # .../polaris/src
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, POLARIS_SRC)

from amplify import AMPLIFY
from amplify.utils.kp_utils.query_utils import grid_queries_nonsquare
from amplify.utils.vis_utils import vis_pred


def load_policy(args) -> AMPLIFY:
    device = torch.device("cuda:0")
    if args.ckpt_path:
        print(f"Loading bundled AMPLIFY from {args.ckpt_path}...")
        policy = AMPLIFY.load(args.ckpt_path, device=device)
    else:
        print("Bundling AMPLIFY from separate checkpoints...")
        print(f"  MT:  {args.mt_ckpt}")
        print(f"  FD:  {args.fd_ckpt}")
        print(f"  ID:  {args.id_ckpt}")
        bundle_save = args.bundle_save
        policy, saved = AMPLIFY.bundle(
            motion_tokenizer_ckpt=args.mt_ckpt,
            forward_dynamics_ckpt=args.fd_ckpt,
            inverse_dynamics_ckpt=args.id_ckpt,
            save_to=bundle_save,
        )
        print(f"Bundled model saved to: {saved}")
    policy.eval()
    return policy


def denormalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """
    Model outputs tanh-squashed min-max normalized actions in (-1, 1).
    Denormalize: a = (out + 1) / 2 * (a_max - a_min) + a_min
    """
    return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplify_root", type=str, default=None, help="Path to AMPLIFY repo root (default: ../../../AMPLIFY relative to this script)")
    # Checkpoint options: either a bundled AMPLIFY ckpt or three separate ones
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to bundled AMPLIFY checkpoint")
    parser.add_argument("--mt_ckpt", type=str, default=None, help="Path to motion tokenizer checkpoint")
    parser.add_argument("--fd_ckpt", type=str, default=None, help="Path to forward dynamics checkpoint")
    parser.add_argument("--id_ckpt", type=str, default=None, help="Path to inverse dynamics checkpoint")
    parser.add_argument("--bundle_save", type=str, default=None, help="Where to save bundled checkpoint (optional)")
    # Task-specific files
    parser.add_argument("--text_emb", type=str, required=True, help="Path to .npy text embedding (1, 512)")
    parser.add_argument("--action_stats", type=str, default=None, help="Path to .npy action stats dict (min, max). Omit if model was trained with normalize_actions: null")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--vis_tracks", action="store_true", help="Predict and return track visualization each step")
    parser.add_argument("--num_tracks", type=int, default=400, help="Target n_tracks passed to grid_queries_nonsquare (default: 400 → 390 actual for 240x426)")
    parser.add_argument("--orig_h", type=int, default=240, help="Original image height used during track preprocessing (default: 240)")
    parser.add_argument("--orig_w", type=int, default=426, help="Original image width used during track preprocessing (default: 426)")
    args = parser.parse_args()

    if not args.ckpt_path and not (args.mt_ckpt and args.fd_ckpt and args.id_ckpt):
        parser.error("Provide either --ckpt_path or all of --mt_ckpt --fd_ckpt --id_ckpt")

    # Load policy
    policy = load_policy(args)

    # Load task artifacts
    text_emb = np.load(args.text_emb)                          # (1, 512) or (512,)
    if text_emb.ndim == 1:
        text_emb = text_emb[None]                               # (1, 512)
    text_emb_t = torch.from_numpy(text_emb).float().to("cuda:0")  # (1, 512)

    if args.action_stats:
        action_stats = np.load(args.action_stats, allow_pickle=True).item()
        action_min = action_stats["min"].astype(np.float32)
        action_max = action_stats["max"].astype(np.float32)
        print(f"Action stats loaded: min={action_min.round(3)}, max={action_max.round(3)}")
    else:
        action_min = action_max = None
        print("No action stats provided — model output used as-is (no denormalization)")

    # Pre-compute grid queries for track visualization
    # Use the same non-square grid as during preprocessing (orig_h x orig_w)
    if args.vis_tracks:
        init_queries = grid_queries_nonsquare(
            views=1, n_tracks=args.num_tracks, device=torch.device("cuda:0"),
            image_height=args.orig_h, image_width=args.orig_w,
        ).standard()
        # init_queries: (1, actual_n_tracks, 2) — aspect-ratio-correct grid
        actual_n = init_queries.shape[1]
        print(f"Track visualization enabled: {actual_n} grid tracks per view ({args.orig_h}x{args.orig_w} aspect ratio)")

    # ZMQ server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"AMPLIFY server listening on port {args.port}")

    while True:
        raw = socket.recv()
        try:
            request = pickle.loads(raw)
        except Exception as e:
            print(f"[Server] Deserialize error: {e} | first 8 bytes: {raw[:8]}")
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))
            continue

        if request.get("reset"):
            socket.send(pickle.dumps({"status": "ok"}))
            continue

        # image: (v, 240, 426, 3) float32 [0,1]
        # proprio: (10,) float32 — pos(3) + rot6d(6) + gripper(1)
        image_np  = request["image"]    # (v, 240, 426, 3)
        proprio_np = request["proprio"]  # (10,)

        image_t  = torch.from_numpy(image_np).float().unsqueeze(0).to("cuda:0")    # (1, v, 240, 426, 3)
        proprio_t = torch.from_numpy(proprio_np).float().unsqueeze(0).to("cuda:0") # (1, 10)

        # text_emb broadcast to batch size 1
        te = text_emb_t  # (1, 512)

        with torch.no_grad():
            actions = policy.act(
                images=image_t,
                proprio=proprio_t,
                text_emb=te,
                ar_sampling="argmax",
            )  # (1, action_horizon, 10) in normalized [-1, 1]

        action_chunk_norm = actions.squeeze(0).cpu().numpy()   # (action_horizon, 10)
        if action_min is not None:
            action_chunk = denormalize_actions(action_chunk_norm, action_min, action_max)
        else:
            action_chunk = action_chunk_norm

        response = {"action_chunk": action_chunk}

        if args.vis_tracks:
            num_views = image_t.shape[1]  # v
            traj_queries = repeat(init_queries, "1 n d -> b v 1 n d", b=1, v=num_views)
            with torch.no_grad():
                pred_traj = policy.predict_traj(
                    images=image_t,
                    init_queries=traj_queries,
                    text_emb=te,
                    ar_sampling="argmax",
                )  # (1, v, t+1, n, 2)
            vis_frame = vis_pred(image_t, pred_traj)  # (1, 128, v*128, 3) uint8
            response["vis_frame"] = vis_frame[0].cpu().numpy()  # (128, v*128, 3)

        socket.send(pickle.dumps(response))


if __name__ == "__main__":
    main()
