"""
Convert collected_data episodes into a LeRobotDataset.

Dataset structure expected:
    collected_data/
        episode_000000/
            trajectory.npz   # keys: traj_obs, traj_cmd, cup_pos
            video.mp4        # agentview RGB
            video_wrist.mp4  # wrist RGB

Usage:
    python create_polaris_dataset.py \
        --data_dir /home/haotian/polaris/collected_data \
        --repo_id haotian/polaris_lerobot \
        --task "pick up the red cup"
"""

import argparse
import os
from glob import glob
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str) -> np.ndarray:
    """Read all frames from an mp4, return (T, H, W, 3) uint8 RGB array."""
    frames = iio.imread(video_path, plugin="pyav")  # (T, H, W, 3) RGB
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Unexpected frame shape from {video_path}: {frames.shape}")
    return frames


def get_video_fps(video_path: str) -> float:
    meta = iio.immeta(video_path, plugin="pyav")
    fps = meta.get("fps", 30.0)
    return float(fps) if fps else 30.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def gen_polaris_dataset(
    data_dir: str,
    repo_id: str,
    task: str,
    img_shape: tuple = (256, 256),
    num_episodes: str = "all"
) -> LeRobotDataset:
    """
    Args:
        data_dir:   path to collected_data/
        repo_id:    HuggingFace repo id for the new dataset
        task:       natural language task description string
        img_shape:  (H, W) to resize frames to; set None to keep original
    """
    episode_dirs = sorted(glob(os.path.join(data_dir, "episode_*")))
    if len(episode_dirs) == 0:
        raise ValueError(f"No episode_* folders found in {data_dir}")

    # ------------------------------------------------------------------ #
    # Infer shapes from the first episode
    # ------------------------------------------------------------------ #
    sample_npz  = np.load(os.path.join(episode_dirs[0], "trajectory.npz"))
    sample_obs  = sample_npz["traj_obs"]   # (T, obs_dim)
    sample_cmd  = sample_npz["traj_cmd"]   # (T, action_dim)

    obs_dim    = sample_obs.shape[-1]
    action_dim = sample_cmd.shape[-1]

    # State = joint obs only
    state_dim = obs_dim

    # Determine image shape from first video
    sample_video = os.path.join(episode_dirs[0], "video.mp4")
    _sample_frames = load_video_frames(sample_video)
    orig_h, orig_w = _sample_frames.shape[1], _sample_frames.shape[2]
    dataset_fps = get_video_fps(sample_video)

    H, W = img_shape if img_shape is not None else (orig_h, orig_w)

    print(f"obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"state_dim={state_dim}, image=({H},{W}), fps={dataset_fps}")
    print(f"Found {len(episode_dirs)} episodes in {data_dir}")

    # ------------------------------------------------------------------ #
    # Feature schema
    # ------------------------------------------------------------------ #
    # Build state feature names
    state_names = [f"joint_obs_{i}" for i in range(obs_dim)]

    action_names = [f"joint_cmd_{i}" for i in range(action_dim)]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
        "observation.images.cam_azure_kinect_front.color": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
    }

    # ------------------------------------------------------------------ #
    # Create dataset
    # ------------------------------------------------------------------ #
    print(f"Creating LeRobotDataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(dataset_fps),
        features=features,
    )

    # ------------------------------------------------------------------ #
    # Process episodes
    # ------------------------------------------------------------------ #
    skipped = 0
    for ep_dir in tqdm(episode_dirs, desc="Episodes"):
        ep_idx = int(ep_dir.split("_")[-1])
        if num_episodes != "all" and ep_idx >= int(num_episodes):
            break
        npz_path    = os.path.join(ep_dir, "trajectory.npz")
        video_path  = os.path.join(ep_dir, "video.mp4")
        wrist_path  = os.path.join(ep_dir, "video_wrist.mp4")

        # Basic existence check
        missing = [p for p in [npz_path, video_path, wrist_path] if not os.path.exists(p)]
        if missing:
            print(f"  Skipping {ep_dir}: missing {missing}")
            skipped += 1
            continue

        try:
            npz = np.load(npz_path)
            traj_obs = npz["traj_obs"].astype(np.float32)  # (T, obs_dim)
            traj_cmd = npz["traj_cmd"].astype(np.float32)  # (T, action_dim)
            # cup_pos  = npz["cup_pos"].astype(np.float32)   # (T, 3) or (3,)

            # If cup_pos is a single vector, broadcast to (T, cup_dim)
            T = traj_obs.shape[0]
            # if cup_pos.ndim == 1:
            #     cup_pos = np.tile(cup_pos[None], (T, 1))  # (T, cup_dim)

            # Concatenate into state
            states = traj_obs
            
            # Load video frames
            agentview_frames = load_video_frames(video_path)   # (T_v, H, W, 3)
            wrist_frames     = load_video_frames(wrist_path)   # (T_v, H, W, 3)

            # Align lengths (video may have 1 extra frame vs trajectory)
            T_min = min(T, len(agentview_frames), len(wrist_frames))
            if T_min < T:
                print(f"  Warning: {ep_dir} traj T={T} > video frames={T_min}, truncating")
            traj_obs = traj_obs[:T_min]
            traj_cmd = traj_cmd[:T_min]
            states   = states[:T_min]
            agentview_frames = agentview_frames[:T_min]
            wrist_frames     = wrist_frames[:T_min]

            # Resize if needed (PIL via imageio's pillow backend)
            if (orig_h, orig_w) != (H, W):
                from PIL import Image
                def resize_frames(frames):
                    return np.stack([
                        np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR))
                        for f in frames
                    ])
                agentview_frames = resize_frames(agentview_frames)
                wrist_frames     = resize_frames(wrist_frames)

            # Add frames
            for t in range(T_min):
                dataset.add_frame({
                    "task":                          task,
                    "observation.state":             states[t],
                    "action":                        traj_cmd[t],
                    "observation.images.cam_azure_kinect_front.color": agentview_frames[t],
                    "observation.images.cam_wrist":     wrist_frames[t],
                })

            dataset.save_episode()

        except Exception as e:
            print(f"  ERROR processing {ep_dir}: {e}")
            skipped += 1
            continue

    print(f"\nDone! {len(episode_dirs) - skipped}/{len(episode_dirs)} episodes saved.")
    print(f"Dataset root: {dataset.root}")
    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert collected_data to LeRobotDataset")
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/haotian/polaris/collected_data",
        help="Path to collected_data folder containing episode_* subdirs",
    )
    parser.add_argument(
        "--repo_id", type=str,
        default="haotian/polaris_lerobot",
        help="HuggingFace repo id (or local name) for the output dataset",
    )
    parser.add_argument(
        "--task", type=str,
        default="pick up the red cup and place it in the target location",
        help="Natural language task description",
    )
    parser.add_argument(
        "--img_height", type=int, default=720,
        help="Resize image height (set 0 to keep original)",
    )
    parser.add_argument(
        "--img_width", type=int, default=1280,
        help="Resize image width (set 0 to keep original)",
    )
    parser.add_argument(
        "--num_episodes", type=str, default="all",
        help="Number of episodes to process (set to 'all' for all episodes)",
    )
    args = parser.parse_args()

    img_shape = (args.img_height, args.img_width) if args.img_height > 0 else None

    gen_polaris_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        task=args.task,
        img_shape=img_shape,
        num_episodes=args.num_episodes,
    )