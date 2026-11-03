import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
import imageio.v3 as iio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--traj_file", type=str, required=True, help="Path to trajectory.npz")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output videos")
parser.add_argument("--mode", type=str, default="joint", choices=["joint", "ee"],
                    help="joint: replay action_joint directly; ee: replay action_ee via IK")
parser.add_argument("--controller", type=str, default="DROID-PutRedCup-no-curtain",
                    help="Environment/controller id passed to gym.make and parse_env_cfg")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import polaris.environments 
from isaaclab_tasks.utils import parse_env_cfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions

CONTROLLER = args_cli.controller
env_cfg = parse_env_cfg(CONTROLLER, device="cuda", num_envs=1, use_fabric=True)
env: ManagerBasedRLSplatEnv = gym.make(CONTROLLER, cfg=env_cfg)

language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

DEVICE = "cuda:0"

from pathlib import Path
output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

traj = np.load(args_cli.traj_file)

frames       = [obs["splat"]["cam0"]]
frames_1     = [obs["splat"]["cam1"]]
wrist_frames = [obs["splat"]["wrist_cam"]]

if args_cli.mode == "joint":
    action_joint = traj["action_joint"]  # (T, 8) = 7 arm joints + 1 gripper
    print(f"Replaying {action_joint.shape[0]} steps using action_joint from {args_cli.traj_file}")

    for i in range(action_joint.shape[0]):
        action = torch.from_numpy(action_joint[i]).float().unsqueeze(0).to(DEVICE)  # (1, 8)
        obs, rew, term, trunc, info = env.step(action, expensive=True)
        frames.append(obs["splat"]["cam0"])
        frames_1.append(obs["splat"]["cam1"])
        wrist_frames.append(obs["splat"]["wrist_cam"])
        print(f"Step: {i}")

else:  # ee mode
    from polaris.utils_.planner_utils import setup_curobo
    from curobo.types.math import Pose

    motion_gen = setup_curobo()

    action_ee = traj["action_ee"]  # (T, 8) = pos(3) + quat_wxyz(4) + gripper(1)
    print(f"Replaying {action_ee.shape[0]} steps using action_ee (via IK) from {args_cli.traj_file}")

    for i in range(action_ee.shape[0]):
        ee_pos  = torch.from_numpy(action_ee[i, :3]).float().unsqueeze(0).to(DEVICE)   # (1, 3)
        ee_quat = torch.from_numpy(action_ee[i, 3:7]).float().unsqueeze(0).to(DEVICE)  # (1, 4) wxyz
        gripper = action_ee[i, 7]

        goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
        ik_result = motion_gen.ik_solver.solve_single(goal_pose)

        if ik_result.success.item():
            joint_solution = ik_result.solution.squeeze(0)[:, :7]  # (1, 7)
            gripper_action = torch.tensor([[gripper]], device=DEVICE)
            action = torch.cat([joint_solution, gripper_action], dim=1)  # (1, 8)
        else:
            print(f"[Step {i}] IK failed, holding current joints")
            action = obs["joint_pos"][:8].unsqueeze(0)

        obs, rew, term, trunc, info = env.step(action, expensive=True)
        frames.append(obs["splat"]["cam0"])
        frames_1.append(obs["splat"]["cam1"])
        wrist_frames.append(obs["splat"]["wrist_cam"])
        print(f"Step: {i}")

frames = np.stack(frames, axis=0)
iio.imwrite(str(output_dir / "replay_cam0.mp4"), frames, fps=30)

frames_1 = np.stack(frames_1, axis=0)
iio.imwrite(str(output_dir / "replay_cam1.mp4"), frames_1, fps=30)

wrist_frames = np.stack(wrist_frames, axis=0)
iio.imwrite(str(output_dir / "replay_wrist.mp4"), wrist_frames, fps=30)

env.close()
simulation_app.close()
