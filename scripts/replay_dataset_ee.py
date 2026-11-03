import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
import imageio.v3 as iio
import numpy as np
import json
from polaris.utils_.planner_utils import setup_curobo

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output videos")
parser.add_argument("--demo_file", type=str, required=True, help="Path to input .npz demo file")
parser.add_argument("--max_steps", type=int, default=None, help="Max steps to replay (default: all)")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions, DATA_PATH
from polaris.utils_.vis_utils import debug_plot
import omni.replicator.core as rep


CONTROLLER = "DROID-PutRedCup-no-curtain"
env_cfg = parse_env_cfg(
    CONTROLLER,
    device="cuda",
    num_envs=1,
    use_fabric=True,
)



def get_cam_param(calibration_path):
    with open(calibration_path, "r") as f:
        cams = json.load(f)

    calibration = {}
    for name, c in cams.items():
        calibration[name] = {
            "intrinsic":  np.array(c["intrinsic"]),
            "extrinsic":  np.array(c["extrinsic"]),  # cam_to_base (4, 4)
            "distortion": np.array(c["distortion"]),
        }

    return calibration


env: ManagerBasedRLSplatEnv = gym.make(CONTROLLER, cfg=env_cfg)



language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
calibration = get_cam_param(str(DATA_PATH / "put_red_cup_no_curtain/cam_calibration.json"))

obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)


from scipy.spatial.transform import Rotation
import numpy as np

motion_gen = setup_curobo()

step = 0
max_steps = 50
frames = [obs["splat"]["cam0"]]
frames_2 = [obs["splat"]["cam1"]]
wrist_frames = [obs["splat"]["wrist_cam"]]

import torch
DEVICE = "cuda:0"

traj = np.load(args_cli.demo_file)

total_steps = traj["frame_indices"].shape[0]
if args_cli.max_steps is not None:
    total_steps = min(total_steps, args_cli.max_steps)
print(f"Replaying {total_steps} steps from {args_cli.demo_file}")

for i in range(total_steps):
    ee_pos = traj["action_pos_right"][i]
    ee_quat = traj["action_orixyzw_right"][i]
    ee_width = traj["action_gripper_right"][i]
    if ee_width > 0:
        ee_width = 0
    else:
        ee_width = 1

    ee_pos = torch.from_numpy(ee_pos).to(DEVICE)
    print("ee_quat_xyzw = ", ee_quat)
    print("ee_pos = ", ee_pos)
    print("ee_width = ", ee_width)
    

    ee_quat = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])  # Convert xyzw to wxyz
    ee_quat = torch.from_numpy(ee_quat).to(DEVICE)

    from curobo.types.math import Pose

    goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
    ik_result = motion_gen.ik_solver.solve_single(goal_pose)

    joint_solution = ik_result.solution.squeeze(0)[:, :8]
    joint_solution[:, -1] = ee_width  # 0=open, 1=close
    

    # action = make_action(target_pos, target_quat, gripper, arm_action)
    obs, rew, term, trunc, info = env.step(joint_solution, expensive=True)
    frames.append(obs["splat"]["cam0"])
    
    frames_2.append(obs["splat"]["cam1"])
    wrist_frames.append(obs["splat"]["wrist_cam"])

    print("Step:", i)



print("Saving videos...")
from pathlib import Path
output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

frames = np.stack(frames, axis=0)
iio.imwrite(str(output_dir / "scene_viz.mp4"), frames, fps=30)
print(f"Saved {output_dir / 'scene_viz.mp4'}")

wrist_frames = np.stack(wrist_frames, axis=0)
iio.imwrite(str(output_dir / "scene_viz_wrist.mp4"), wrist_frames, fps=30)
print(f"Saved {output_dir / 'scene_viz_wrist.mp4'}")

frames_2 = np.stack(frames_2, axis=0)
iio.imwrite(str(output_dir / "scene_viz_2.mp4"), frames_2, fps=30)
print(f"Saved {output_dir / 'scene_viz_2.mp4'}")

env.close()
simulation_app.close()