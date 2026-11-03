import torch
from scipy.spatial.transform import Rotation
import numpy as np
import os

from polaris.utils_.data_utils import ObsRecorder
from polaris.utils_.eval_utils import is_success

def setup_curobo(robot_cfg="franka_robotiq_2f_85.yml"):

    import omni.usd
    from curobo.util.usd_helper import UsdHelper
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    stage = omni.usd.get_context().get_stage()
    assert stage is not None

    robot_prim_path = "/World/envs/env_0/robot"
    usd_help = UsdHelper()
    usd_help.stage = stage

    obstacle_world = usd_help.get_obstacles_from_stage(
        only_paths=["/World/envs/env_0"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,                       
            "/World/defaultGroundPlane",
            "randomization",        
            "workspace_static",                   
            "OmniverseKitViewportCameraMesh",   
            "CameraModel",                        
            "env_light",                    
            "defaultLight",                 
            "Environment",                         
            "Render",
        ],
    )
    print("Obstacle meshes:", [m.name for m in (obstacle_world.mesh or [])])

    world_cfg = obstacle_world.get_collision_check_world()
    tensor_args = TensorDeviceType()

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        interpolation_dt=0.02,
        use_cuda_graph=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    return motion_gen

class MotionPlanner:
    def __init__(
        self,
        env,
        waypoints: list,
        recorder: ObsRecorder,
        device: str = "cuda",
        robot_cfg: str = "franka_robotiq_2f_85",
        debug: bool = False,
    ):
        self.env  = env
        self.waypoints = waypoints
        self.device = device
        self.recorder = recorder
        self.debug = debug
        self.motion_gen  = setup_curobo(f"{robot_cfg}.yml")

        self.GRIPPER_JOINT_IDX = 7
        self.GRIPPER_OPEN_VAL = 0
        self.GRIPPER_CLOSE_VAL = 1

        self.FINGER_CLOSED_Y = 0
        self.FINGER_OPEN_Y = 0.05

        self.GRIPPER_OPEN_ENV = 0.7

    def reset(self, object_poses):
        obs, info = self.env.reset(object_positions=object_poses)
        self.recorder.reset()
        return obs
    
    
    def get_current_joints(self, obs) -> torch.Tensor:
        """Return (13,) joint positions on CUDA from policy obs dict."""

        return obs["joint_pos"]  # (13,)
 
    def plan_trajectory(self, current_joints, target_pos, target_quat):

        from curobo.types.math import Pose
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        from curobo.types.state import JointState

        goal_pose = Pose(position=target_pos, quaternion=target_quat)

        curobo_joints = torch.cat([
            current_joints[:self.GRIPPER_JOINT_IDX],
            torch.zeros(current_joints.shape[0] - self.GRIPPER_JOINT_IDX, device=self.device)
        ])

        start_state = JointState.from_position(
            curobo_joints.unsqueeze(0),
            joint_names=self.motion_gen.joint_names,
        )

        result = self.motion_gen.plan_single(
            start_state=start_state,
            goal_pose=goal_pose,
            plan_config=MotionGenPlanConfig(max_attempts=5, time_dilation_factor=0.8),
        )
        if result.success[0]:
            return result.get_interpolated_plan().position[:, :self.GRIPPER_JOINT_IDX]  # (T, 7) arm joints only
        else:
            print(f"[cuRobo] Planning failed! status={result.status}")
            if result.position_error is not None:
                print(f"  position_error={result.position_error}")
            
        print("Use IK")
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
        print("IK success:", ik_result.success)
        print("IK solution:", ik_result.solution)
        print("IK error:", ik_result.error)

        if ik_result.success.item():
            return ik_result.solution.squeeze(0)[:, :self.GRIPPER_JOINT_IDX]
        else:
            return None
        
    def action_ee(self, action, gripper_val):
        """Convert arm joint action to end-effector pose + gripper command."""
        from curobo.types.robot import JointState

        action_ee = self.motion_gen.rollout_fn.compute_kinematics(
            JointState.from_position(action[:self.GRIPPER_JOINT_IDX])
        )
        ee_pos = action_ee.ee_pos_seq.squeeze(0)
        ee_quat = action_ee.ee_quat_seq.squeeze(0)

        gripper_action = torch.tensor([gripper_val], device=self.device)
        return torch.cat([ee_pos, ee_quat, gripper_action]).unsqueeze(0) # (1, 8)
 
    def execute_trajectory(self, obs, traj, gripper_val):
        """
        Execute trajectory by stepping through waypoints.

        Args:
            env: Isaac environment
            obs: Current observation
            traj: Joint trajectory from cuRobo (T, 7)
            gripper_val: Gripper command (0=open, 1=close)
        """
        print(f"  Executing {traj.shape[0]} waypoints")
        gripper_action = torch.tensor([gripper_val], device=self.device)

        for i in range(traj.shape[0]):
            action = traj[i].clone()
            obs["policy"]["action_ee"]  = self.action_ee(action, gripper_val)
            action = torch.cat([action, gripper_action])  # (8,) = (7 arm joints + 1 gripper)
            obs["policy"]["action_joint"]  = action.unsqueeze(0)  # (1, 8)
            
            self.add_obs(obs)

            if i == 0:
                print(f"  [First waypoint] cuRobo traj[0]: {traj[i]}")
                print(f"  [First waypoint] action: {action}")
            if i == traj.shape[0] - 1:
                print(f"  [Last waypoint] cuRobo traj[-1]: {traj[i]}")
                print(f"  [Last waypoint] action: {action}")
           
            obs, rew, term, trunc, info = self.env.step(action.unsqueeze(0), expensive=True)
            
            if term[0] or trunc[0]:
                return obs, info, True
    
        return obs, info, False
    
    def add_obs(self, obs):
        self.recorder.add(obs)

    def subgoal_gripper_state(self, obs, new_grasp: float, n_steps: int = 2):
        """Hold current arm joints and only change the gripper for n_steps."""
        joints = self.get_current_joints(obs)                     # (13,)
        arm_joints    = joints[:self.GRIPPER_JOINT_IDX]           # (7,)
        gripper_action = torch.tensor([new_grasp], device=self.device)
        action = torch.cat([arm_joints, gripper_action])          # (8,)

        print(f"  [Gripper change] target={'CLOSE' if new_grasp else 'OPEN'} over {n_steps} steps")
        for _ in range(n_steps):

            obs["policy"]["action_ee"]  = self.action_ee(arm_joints, new_grasp)
            obs["policy"]["action_joint"]  = action.unsqueeze(0)  # (1, 8)
            self.add_obs(obs)
            obs, rew, term, trunc, info = self.env.step(action.unsqueeze(0), expensive=True)
            if term[0] or trunc[0]:
                return obs, info, True

        return obs, info, False

    def execute_waypoints(self, obs, object_poses: dict):
        """
        Execute a sequence of waypoints defined relative to object poses.
        """
        info = {}
        
        for i, wp in enumerate(self.waypoints):
            obj = wp["object"]
            offset = wp["offset"]
            rel_quat_xyzw = wp["rel_quat"]  # x y z w
            grasp = wp["grasp"]
            subgoal = wp["subgoal"]

            if "/" in obj:
                obj_names = obj.split("/")
                poses = [object_poses[o] for o in obj_names]
                # midpoint of positions
                obj_pos = [
                    sum(p[k] for p in poses) / len(poses)
                    for k in range(3)
                ]
                # use identity orientation for midpoint waypoints
                obj_quat_wxyz = object_poses[obj_names[0]][3:]  # Use the orientation of the first object
                print(f"\n[Waypoint {i}] obj=MIDPOINT({obj_names}) grasp={grasp}")
            else:
                obj_pose = object_poses[obj] # [x, y, z, qw, qx, qy, qz]
                obj_pos = obj_pose[:3]
                obj_quat_wxyz = obj_pose[3:] # [qw, qx, qy, qz]
                print(f"\n[Waypoint {i}] obj={obj} grasp={grasp}")

            target_pos, target_quat = self.compute_waypoint_pose(
                obj_pos, obj_quat_wxyz, offset, rel_quat_xyzw,
            )

            print(f"\n[Waypoint {i}] obj={obj} grasp={grasp}")
            print(f"  target pos:  {target_pos}")
            print(f"  target quat: {target_quat}")
            traj = self.plan_trajectory(self.get_current_joints(obs), target_pos, target_quat)

            if traj is not None:
                obs, info, done = self.execute_trajectory(
                    obs, traj, grasp
                )
                
                if subgoal:
                    toggled_grasp = 1 - grasp
                    print(f"  [Subgoal] toggling gripper {grasp} → {toggled_grasp}")
                    obs, info, done = self.subgoal_gripper_state(obs, toggled_grasp)
                    self.recorder.save_subgoal()

                if done:
                    print(f"  Early termination at waypoint {i}.")
                    self.add_obs(obs) # record terminal state ONCE here
                    return obs, info, True
            else:
                print(f"  Planning failed at waypoint {i}.")
                if self.debug:
                    self.add_obs(obs)
                    self.recorder.save_episode()
                    print("Save visualization for debug")
                return obs, info, False
        
        # record terminal state ONCE after all waypoints complete
        self.add_obs(obs)

        if is_success(info) or self.debug:
            self.recorder.save_episode()
            return obs, info, True

        return obs, info, False
 
    def compute_waypoint_pose(self, obj_pos, obj_quat_wxyz, offset, rel_quat_wxyz):
        """
        Compute absolute world-frame pose for a waypoint.
        obj_pos: [x, y, z]
        obj_quat_wxyz: [qw, qx, qy, qz] - object orientation in world
        offset: [dx, dy, dz] - offset in world frame
        rel_quat_wxyz: [w, x, y, z] - relative rotation to apply on top of object orientation
        """

        # object orientation
        obj_R = Rotation.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])

        # absolute position = object pos + offset
        pos = torch.tensor([
            obj_pos[0] + offset[0],
            obj_pos[1] + offset[1],
            obj_pos[2] + offset[2],
        ], device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)

        # absolute orientation = obj_orientation * rel_quat
        rel_R = Rotation.from_quat([rel_quat_wxyz[1], rel_quat_wxyz[2], rel_quat_wxyz[3], rel_quat_wxyz[0]])  # xyzw
        abs_R = obj_R * rel_R
        q = abs_R.as_quat()  # xyzw
        quat = torch.tensor([q[3], q[0], q[1], q[2]], device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, 4) wxyz

        return pos, quat