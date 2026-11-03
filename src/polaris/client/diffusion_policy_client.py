import pickle
import numpy as np
import cv2
import zmq
import torch

import robosuite.utils.transform_utils as T

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs
from polaris.utils_.transform_utils import (
    state_quat_to_rot6d,
    rotation_transfer_6D_to_matrix,
)

IMAGE_H = 240
IMAGE_W = 426
DEVICE = "cuda:0"


@InferenceClient.register(client_name="DiffusionPolicy")
class DiffusionPolicyClient(InferenceClient):
    """
    Client for DiffusionUnetHybridImagePolicy served by diffusion_policy_server.py.

    Policy input:
        image:     (B, T, 3, 240, 426)
        agent_pos: (B, T, 10)  = pos(3) + rot6d(6) + gripper(1)

    Policy output:
        action chunk: (n_action_steps, 10)  = pos(3) + rot6d(6) + gripper(1) absolute EEF

    Env input (polaris):
        (1, 8) = 7 arm joint positions + 1 gripper  (joint position controller)
    """

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        host = args.host if args.host is not None else "localhost"
        port = args.port if args.port is not None else 5556
        open_loop_horizon = args.open_loop_horizon
        if open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DiffusionPolicyClient")
        self.open_loop_horizon = open_loop_horizon

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to diffusion policy server at {host}:{port}")

        self.action_chunk = None
        self.actions_from_chunk_completed = 0

        # Setup curobo IK solver for EEF pose → joint positions
        from polaris.utils_.planner_utils import setup_curobo
        self.motion_gen = setup_curobo()

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.action_chunk = None
        self.actions_from_chunk_completed = 0
        self.socket.send(pickle.dumps({"reset": True}))
        self.socket.recv()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        image, agent_pos, state = self._extract_observation(obs)

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            # Request new chunk from server
            request = {"image": image, "agent_pos": agent_pos}
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            self.action_chunk = response["action_chunk"]  # (n_action_steps, 10)
            self.actions_from_chunk_completed = 0

        action10 = self.action_chunk[self.actions_from_chunk_completed]  # (10,)
        self.actions_from_chunk_completed += 1

        env_action = self._policy_action_to_env_action(action10, obs)  # (8,)

        viz = None
        if return_viz:
            viz_raw = obs["splat"]["cam1"]  # same camera as policy input
            viz = cv2.resize(viz_raw, (IMAGE_W, IMAGE_H))  # save at policy input resolution (426, 240)

        return env_action, viz

    def _extract_observation(self, obs_dict: dict):
        # Image: resize exterior cam to policy input resolution
        raw_image = obs_dict["splat"]["cam1"]  # cam1 is the front camera in front of the robot, (H, W, 3) uint8
        image = cv2.resize(raw_image, (IMAGE_W, IMAGE_H))  # (240, 426, 3) uint8

        # State: pos(3) + quat_wxyz(4) + gripper(1)  →  pos(3) + rot6d(6) + gripper(1)
        state = obs_dict["policy"]["ee_pose"][0].cpu().numpy()  # (8,) numpy
        agent_pos = state_quat_to_rot6d(state)  # (10,) float32

        return image, agent_pos, state

    def _policy_action_to_env_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
        """
        Convert policy action (10,) = pos(3) + rot6d(6) + gripper(1)
        to env action (8,) = 7 arm joint positions + 1 gripper
        via curobo IK.
        """
        from curobo.types.math import Pose

        pos   = action[:3]
        rot6d = action[3:9]
        gripper = action[9]

        # rot6d → rotation matrix → quat xyzw → wxyz
        R = rotation_transfer_6D_to_matrix(rot6d)       # (3, 3)
        quat_xyzw = T.mat2quat(R)                        # (4,) xyzw
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


        ee_pos  = torch.from_numpy(pos).float().unsqueeze(0).to(DEVICE)       # (1, 3)
        ee_quat = torch.from_numpy(quat_wxyz).float().unsqueeze(0).to(DEVICE) # (1, 4) wxyz

        goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)

        if ik_result.success.item():
            joint_pos = ik_result.solution.squeeze(0)[0, :7].cpu().numpy()  # (7,)
        else:
            # IK failed: hold current joint positions
            print("[DiffusionPolicyClient] IK failed, holding current joints")
            joint_pos = obs["policy"]["arm_joint_pos"][0].cpu().numpy()  # (7,)

        # Binarize gripper: 0=open, 1=close
        gripper_bin = 1.0 if gripper > 0.5 else 0.0

        return np.concatenate([joint_pos, [gripper_bin]]).astype(np.float32)  # (8,)
