import os
import sys
import tyro
import argparse

import torch
import numpy as np
import imageio.v3 as iio
import gymnasium as gym


from isaaclab.app import AppLauncher
from polaris.config import DataArgs
from scipy.spatial.transform import Rotation

from polaris.utils_.data_utils import ObsRecorder
from polaris.utils_.planner_utils import MotionPlanner
from polaris.utils_.vis_utils import debug_plot
from polaris.utils_.eval_utils import randomize_object_poses, is_success


def main(data_args: DataArgs):
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = data_args.headless
    args_cli.gpu_id = data_args.gpu_id
    # Force Vulkan renderer to use the same GPU (bypasses Xorg default GPU)
    sys.argv += [f"--/renderer/activeGpu={data_args.gpu_id}"]
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import polaris.environments
    from isaaclab_tasks.utils import parse_env_cfg
    from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
    from polaris.utils import load_eval_initial_conditions, load_task_config
    
    if data_args.env_folder is None:
        data_args.env_folder = os.path.dirname(gym.spec(data_args.environment).kwargs["usd_file"])
        
    DEVICE = data_args.device

    env_cfg = parse_env_cfg(data_args.environment, device=DEVICE, num_envs=1, use_fabric=True)

    env_cfg.episode_length_s = data_args.max_episode_length * (env_cfg.sim.dt * env_cfg.decimation)

    env: ManagerBasedRLSplatEnv = gym.make(data_args.environment, cfg=env_cfg, robot_config=data_args.robot)

    language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
    task_config_path = data_args.task_config or os.path.join(data_args.env_folder, "task_config.yaml")
    object_randomization, waypoints = load_task_config(task_config_path)

    calibration_path = os.path.join(data_args.env_folder, "cam_calibration.json")

    print("Language instruction:", language_instruction)
    print(f"Loaded {len(waypoints)} waypoints from {task_config_path}")

    save_dir = data_args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Resume from last saved episode
    # Find next available episode index
    ep_idx = 0
    while os.path.exists(os.path.join(save_dir, f"episode_{ep_idx:06d}")):
        ep_idx += 1
    num_success = ep_idx
    print(f"Resuming from episode {ep_idx}.")

    num_attempts = 0

    recorder = ObsRecorder(calibration_path, save_dir, fps=30, ep_idx=num_success, debug=data_args.debug)

    planner = MotionPlanner(
        env=env,
        recorder=recorder,
        robot_cfg=data_args.robot,
        waypoints=waypoints,
        device=DEVICE,
        debug=data_args.debug,
    )
    
    while num_success < data_args.num_episodes and num_attempts < data_args.max_attempts:
        num_attempts += 1
        print(f"\n{'='*60}")
        print(f"Attempt {num_attempts:3d} | Successes {num_success:3d}/{data_args.num_episodes}")
        print(f"{'='*60}")

        # Randomize object poses
        object_poses = randomize_object_poses(object_randomization, initial_conditions)[0]

        obs = planner.reset(object_poses)
    
        # Execute waypoints
        obs, info, done = planner.execute_waypoints(obs, object_poses=object_poses)

        # Check success
        print(info)
        if is_success(info):
            print(f"\n  ✓ SUCCESS  (episode {num_success + 1}/{data_args.num_episodes})")
            num_success += 1
            recorder.ep_idx += 1
        else:
            reason = "Metric not satisfied" if not done else "Debugging"
            print(f"\n  ✗ FAILED  ({reason})")

    print(f"\nCollection complete: {num_success} successes in {num_attempts} attempts.")
    print(f"Data saved to: {save_dir}")
    env.close()
    simulation_app.close()



if __name__ == "__main__":
    args: DataArgs = tyro.cli(DataArgs)
    main(args)
