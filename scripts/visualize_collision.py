import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import gymnasium as gym
import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions

from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.types.state import JointState
from curobo.geom.types import Sphere

ENV_NAME  = "DROID-PutRedCup-no-curtain"
ROBOT_CFG = "franka_robotiq_2f_85.yml"
DEVICE    = "cuda"

env_cfg = parse_env_cfg(ENV_NAME, device=DEVICE, num_envs=1, use_fabric=True)
env: ManagerBasedRLSplatEnv = gym.make(ENV_NAME, cfg=env_cfg)
_, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, _ = env.reset(object_positions=initial_conditions[0])

stage          = omni.usd.get_context().get_stage()
usd_help       = UsdHelper()
usd_help.stage = stage

obstacle_world = usd_help.get_obstacles_from_stage(
    only_paths=["/World/envs/env_0"],
    reference_prim_path="/World/envs/env_0/robot",
    ignore_substring=[
        "/World/envs/env_0/robot",
        "/World/defaultGroundPlane",
        "randomization", "external_cam",
        "OmniverseKitViewportCameraMesh", "CameraModel",
        "env_light", "defaultLight", "Environment", "Render",
    ],
)

print("Meshes:",  [m.name for m in (obstacle_world.mesh   or [])])
print("Cuboids:", [c.name for c in (obstacle_world.cuboid or [])])

world_cfg   = obstacle_world.get_collision_check_world()
tensor_args = TensorDeviceType()
motion_gen  = MotionGen(MotionGenConfig.load_from_robot_config(ROBOT_CFG, world_cfg, tensor_args))
motion_gen.warmup()

joints = torch.cat([obs["policy"]["arm_joint_pos"][0], torch.zeros(6, device=DEVICE)])
js     = JointState.from_position(joints.unsqueeze(0), joint_names=motion_gen.joint_names)

spheres = motion_gen.kinematics.get_robot_as_spheres(js.position)
print(f"[DEBUG] Drew {len(spheres[0])} collision spheres")

print(type(spheres))
print(type(spheres[0]))
print(spheres[0])

for i, sphere in enumerate(spheres[0]):
    pos = sphere.position  # [x, y, z]
    r   = float(sphere.radius)
    usd_help.add_sphere_to_stage(
        obstacle=Sphere(
            name=f"sphere_{i}",
            pose=pos + [1, 0, 0, 0],  # wxyz identity
            radius=r,
        ),
        base_frame="/World/curobo_spheres",
    )

from pxr import UsdGeom
robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")
xform = UsdGeom.Xformable(robot_prim)
world_tf = xform.ComputeLocalToWorldTransform(0)
trans = world_tf.ExtractTranslation()
robot_offset = [trans[0], trans[1], trans[2]]
print("Robot base world pos:", robot_offset)

print("Done — inspect in Isaac Sim GUI")

while simulation_app.is_running():
    env.sim.render()