import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import (
    ManagerBasedRLSplatEnv,
)
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import rubric system
from polaris.environments.rubrics import Rubric
from polaris.utils import DATA_PATH
import polaris.environments.rubrics.checkers as checkers


# =============================================================================
# Environment Registration
# =============================================================================

gym.register(
    id="DROID-PutRedCup-no-curtain",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "put_red_cup_no_curtain/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("red_cup", threshold=0.2),
                (checkers.lift("red_cup", threshold=0.04), [0]),
                (checkers.is_within_xy("red_cup", "blue_plate", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-StackBlock",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "stack_block/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("yellow_block", threshold=0.02),
                (checkers.lift("yellow_block", threshold=0.04), [0]),
                checkers.reach("green_block", threshold=0.02),
                (checkers.is_within_xy("yellow_block", "green_block", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-InsertDonut",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "insert_donut/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("blue_donut", threshold=0.2),
                (checkers.lift("blue_donut", threshold=0.15), [0]),
                (checkers.is_within_xy("blue_donut", "bar", percent_threshold=0.1, open_finger_threshold=2), [1]),
                (checkers.is_inserted("blue_donut", "bar", xy_threshold=0.06, z_threshold=0.1), [2]),
            ]
        ),
    },
)

gym.register(
    id="DROID-HangMug",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "hang_mug/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("red_mug", threshold=0.2),
                (checkers.lift("red_mug", threshold=0.05), [0]),
                (checkers.is_close_xy("red_mug", "mug_tree", xy_dist_threshold=0.25), [1]),
                (checkers.is_hung("red_mug", "mug_tree", xy_threshold=0.15, z_threshold=0.1, stable_steps=3, vel_threshold=0.05), [2]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PickPlaceToys",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "pick_place_toys/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.is_within_xy("blue_toy_large", "pink_box", percent_threshold=1),
                checkers.is_within_xy("orange_toy", "pink_box", percent_threshold=1),
                checkers.is_within_xy("yellow_toy", "pink_box", percent_threshold=1),
            ]
        ),
    },
)

gym.register(
    id="DROID-StackBowls",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "stack_bowls/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("green_bowl", threshold=0.2),
                (checkers.lift("green_bowl", threshold=0.04), [0]),
                checkers.reach("blue_bowl", threshold=0.2),
                (checkers.is_within_xy("blue_bowl", "green_bowl", percent_threshold=0.5), [1]),
            ]
        ),
    },
)
