import numpy as np
import asyncio
import json
import websockets
import torch

import msgpack
import msgpack_numpy as m
m.patch()

from polaris.client.abstract_client import InferenceClient, PolicyArgs
from polaris.utils_.planner_utils import setup_curobo
import time


@InferenceClient.register(client_name="point_policy")
class PointPolicyClient(InferenceClient):

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.uri = f"ws://{args.host}:{args.port}"
        self.open_loop_horizon = args.open_loop_horizon
        self.pred_action_chunk = None
        self.motion_gen = setup_curobo()
        self.device = args.device
    
    def ee_to_joint(self, action_ee):
        ee_pos = action_ee[:3]
        ee_quat = action_ee[3:7] # wxyz
        gripper = action_ee[-1]

        ee_pos = torch.from_numpy(ee_pos).to(self.device)
        ee_quat = torch.from_numpy(ee_quat).to(self.device)

        from curobo.types.math import Pose

        goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
        joint_solution = ik_result.solution.squeeze(0)[:, :8]
        joint_solution[:, -1] = gripper
        
        return joint_solution # (1, 8)

    def reset(self):
        self.pred_action_chunk = None
        asyncio.run(self._send({
            "command": "reset",
        }))

    def _serialize(self, v, _path=""):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        if isinstance(v, dict):
            return {k: self._serialize(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._serialize(i) for i in v]
        return v

    async def _send(self, message: dict) -> dict:
        async with websockets.connect(self.uri, max_size=100 * 1024 * 1024) as ws:
            packed = msgpack.packb(message, use_bin_type=True)
            await ws.send(packed)
            response = await ws.recv()
            return msgpack.unpackb(response, raw=False)

    def infer(
        self, obs: dict, language_instruction=None, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        
        result = asyncio.run(self._send({
            "command": "infer",
            "obs": self._serialize(self._extract_observation(obs)),
            "return_viz": return_viz,
        }))

        action_ee = np.array(result["action"])
        
        action = self.ee_to_joint(action_ee)

        action = action.detach().cpu().numpy()
        self._viz = np.array(result["viz"]) if result.get("viz") is not None else None

        return action, self._viz if return_viz else None
    
    def _extract_observation(self, obs_dict):
        # Assign images
        pixels1 = obs_dict["splat"]["cam0"]
        pixels2 = obs_dict["splat"]["cam1"]

        depth1 = obs_dict["splat"]["cam0_depth"]
        depth2 = obs_dict["splat"]["cam1_depth"]

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        prorio = robot_state["ee_pose"]
        gripper_pcd = robot_state["gripper_pcd"]
        states_ee = robot_state["ee_pose"]

        return {
            "pixels1": pixels1,
            "pixels2": pixels2,
            "depth1": depth1,
            "depth2": depth2,
            "prorio": prorio,
            "gripper_pcd": gripper_pcd,
            "states_ee": states_ee,
        }
    
    def shutdown(self):
        """Send shutdown signal to THIS server (by port)."""
        try:
            asyncio.run(self._send({"command": "shutdown"}))
            print(f"Shutdown signal sent to {self.uri}")
        except Exception as e:
            print(f"Server {self.uri} closed: {e}")
