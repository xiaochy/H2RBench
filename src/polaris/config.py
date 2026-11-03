"""
Lightweight config dataclasses for evaluation.
No heavy dependencies - safe to import anywhere.
"""

from dataclasses import dataclass
import gymnasium as gym
import os


@dataclass
class PolicyServer:
    """
    Configuration for a policy server to co-launch.

    Use {port} placeholder in command - it will be replaced with an auto-assigned free port.
    Jobs using this server will automatically have their policy.port updated.

    Example:
        PolicyServer(
            name="pi0",
            command="CUDA_VISIBLE_DEVICES=0 python serve_policy.py --port {port}",
        )
    """

    name: str  # Friendly name for logging (also used to match jobs to servers)
    command: str  # Shell command with {port} placeholder
    ready_message: str = (
        "Application startup complete"  # Message indicating server is ready
    )

    # Runtime-assigned (don't set manually)
    _assigned_port: int | None = None


@dataclass
class PolicyArgs:
    """Policy configuration."""

    # name: str                              # Policy name (pi05_droid_jointpos, pi0_fast_droid_jointpos, etc.)
    client: str = "DroidJointPos"  # Client name (DroidJointPos, Fake, etc.)
    host: str = "localhost"
    port: int = 8000
    open_loop_horizon: int | None = 8
    policy_path: tuple[str, ...] | str = None
    device : str = "cuda:0"
    config_path: str | None = None
    dataset_meta: str | None = None
    cam_key: str = "cam1"
    obs_horizon: int = 2
    render_gripper_only: bool = False


@dataclass
class EvalArgs:
    """Evaluation configuration."""

    policy: PolicyArgs  # Policy arguments
    environment: str  # Which IsaacLab environment to use
    run_folder: str  # Path to run folder
    headless: bool = True  # Whether to run in headless mode
    initial_conditions_file: str | None = None  # Path to initial conditions file
    instruction: str | None = None  # Override language instruction
    rollouts: int | None = None  # Number of rollouts to evaluate
    robot: str = "franka_robotiq_2f_85"
    env_folder: str | None = None
    task_config: str | None = None  # Path to a custom task_config.yaml (overrides env_folder/task_config.yaml)
    device: str = "cuda:0"
    max_episode_length: int = 300 # max_episode_length = episode_length_s / (dt * decimation)
    tqdm_position: int = 0
    seed: int = 42


@dataclass
class DataArgs:
    """Data Generation configuration."""
    environment: str = "DROID-PutRedCup-no-curtain" # Which IsaacLab environment to use
    env_folder: str | None = None
    task_config: str | None = None  # Path to a custom task_config.yaml (overrides env_folder/task_config.yaml)
    save_dir: str | None = None # Path to run folder
    headless: bool = True  # Whether to run in headless mode
    robot: str = "franka_robotiq_2f_85" # Which robot
    instruction: str | None = None  # Override language instruction
    num_episodes: int = 50
    max_attempts: int = 100
    max_episode_length: int = 300  # max_episode_length = episode_length_s / (dt * decimation)
    debug: bool = False
    device: str = "cuda"
    gpu_id: int = 0



@dataclass
class JobCfg:
    """A single evaluation job in a batch."""

    eval_args: EvalArgs
    server: PolicyServer | None = None  # Server to co-launch for this job


@dataclass
class BatchConfig:
    """Batch evaluation configuration."""

    jobs: list[JobCfg]

    # @staticmethod # let users do this on their own if they want
    # def sweep(**kwargs: list[Any]) -> list[dict[str, Any]]:
    #     """
    #     Helper to generate grid of configs from lists of values.

    #     Example:
    #         BatchConfig.sweep(
    #             usd=["env1.usd", "env2.usd"],
    #             policy=["pi0", "pi05"],
    #         )
    #         # Returns 4 dicts: all combinations
    #     """
    #     keys = list(kwargs.keys())
    #     values = [v if isinstance(v, list) else [v] for v in kwargs.values()]
    #     return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
