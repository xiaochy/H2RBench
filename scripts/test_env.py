import argparse

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import polaris.environments  # noqa: F401  (registers the gym envs)
from isaaclab_tasks.utils import parse_env_cfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions

MAX_STEPS = 50

env_cfg = parse_env_cfg(
    "DROID-scene",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)

env: ManagerBasedRLSplatEnv = gym.make("DROID-scene", cfg=env_cfg)

_, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

# Hold a fixed end-effector pose: [pos(3), quat_wxyz(4), gripper(1)]
ee_pose = torch.tensor(
    [[0.35970, 0.0, 0.31842, 0.0, 0.0, 0.0, 1.0, 0.0]],
    device="cuda:0",
    dtype=torch.float32,
)

frames = {
    "scene_viz_cam0.mp4": [obs["splat"]["cam0"]],
    "scene_viz_cam1.mp4": [obs["splat"]["cam1"]],
    "scene_viz_wrist.mp4": [obs["splat"]["wrist_cam"]],
}
cam_keys = {"scene_viz_cam0.mp4": "cam0", "scene_viz_cam1.mp4": "cam1", "scene_viz_wrist.mp4": "wrist_cam"}

for step in range(MAX_STEPS):
    obs, rew, term, trunc, info = env.step(ee_pose, expensive=True)
    print(obs["policy"]["ee_pose"][0])
    for path, cam in cam_keys.items():
        frames[path].append(obs["splat"][cam])

print(f"\nCollected {MAX_STEPS} samples.")

for path, buf in frames.items():
    iio.imwrite(path, np.stack(buf, axis=0), fps=30)

env.close()
simulation_app.close()
