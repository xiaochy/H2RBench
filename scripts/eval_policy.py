import os
import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs
from polaris.utils_.eval_utils import randomize_object_poses, set_seed
import numpy as np

def main(eval_args: EvalArgs):
    set_seed(eval_args.seed)
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions, load_task_config
    from polaris.client import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    if eval_args.env_folder is None:
        eval_args.env_folder = os.path.dirname(gym.spec(eval_args.environment).kwargs["usd_file"])

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )

    env_cfg.episode_length_s = eval_args.max_episode_length * (env_cfg.sim.dt * env_cfg.decimation)
    
    env: ManagerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    task_config_path = eval_args.task_config or os.path.join(
        eval_args.env_folder, "task_config.yaml"
    )
    object_randomization, _ = load_task_config(task_config_path)

    # Randomise object poses
    ic = randomize_object_poses(object_randomization, initial_conditions)[0]

    rollouts = eval_args.rollouts
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    # Fast-forward RNG to match completed episodes so randomization resumes in sequence
    for _ in range(episode):
        randomize_object_poses(object_randomization, initial_conditions)

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = eval_args.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=ic, expensive=True
    )
    policy_client.reset()

    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction, return_viz=True)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=True
        )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=15)

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            episode += 1
            bar = tqdm.tqdm(range(horizon))
            ic = randomize_object_poses(object_randomization, initial_conditions)[0]
            obs, info = env.reset(
                object_positions=ic, expensive = True
            )

            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
