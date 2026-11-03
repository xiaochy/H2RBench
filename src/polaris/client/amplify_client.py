import pickle
import numpy as np
import cv2
import zmq
import torch
from scipy.spatial.transform import Rotation

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs

DEVICE = "cuda:0"

# Default model input size — overridden at runtime from checkpoint config if policy_path is provided
_DEFAULT_IMAGE_H = 240
_DEFAULT_IMAGE_W = 426

# Visualization output resolution — each view at native camera resolution, two views stitched side by side
VIZ_H = 240
VIZ_W = 426 * 2  # 852 — both views at full resolution


def _read_img_shape_from_ckpt(ckpt_path: str):
    """Load img_shape from checkpoint's motion_tokenizer_cfg. Returns (H, W) or None."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        img_shape = ckpt["config"]["motion_tokenizer_cfg"]["img_shape"]
        return int(img_shape[0]), int(img_shape[1])
    except Exception as e:
        print(f"[AMPLIFYFullResClient] Could not read img_shape from ckpt ({e}), using default {_DEFAULT_IMAGE_H}×{_DEFAULT_IMAGE_W}")
        return None


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz → (6,) rot6d (first two columns of rotation matrix)."""
    xyzw = quat_wxyz[[1, 2, 3, 0]]               # wxyz → xyzw
    R = Rotation.from_quat(xyzw).as_matrix()      # (3, 3)
    return R[:, :2].T.flatten().astype(np.float32) # (6,) = [col0, col1] row-major


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """(6,) rot6d → (4,) wxyz quaternion via Gram-Schmidt."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)            # (3, 3) columns
    xyzw = Rotation.from_matrix(R).as_quat()      # (4,) xyzw
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)  # wxyz


@InferenceClient.register(client_name="AMPLIFY")
class AMPLIFYFullResClient(InferenceClient):
    """
    Client for a full-resolution AMPLIFY variant that accepts 240×426 images directly
    (no resize to 128×128).

    Policy input  (sent to server):
        image:   (v=2, 240, 426, 3) float32 [0, 1]
        proprio: (10,) float32 = pos(3) + rot6d(6) + gripper(1)

    Policy output (received from server):
        action_chunk: (action_horizon, 10) float32
                      = pos(3) + rot6d(6) + gripper(1) in robot base frame,
                      denormalized back to original EEF pose space.

    Env input (polaris):
        (1, 8) = 7 arm joint positions + 1 gripper
        Converted from EEF pose via: rot6d → quat_wxyz → curobo IK.

    Args in PolicyArgs:
        host:              ZMQ server host (default: localhost)
        port:              ZMQ server port (default: 5557)
        open_loop_horizon: steps to execute per chunk before re-querying (default: 1)
        policy_path:       path to bundled AMPLIFY .pt — used to read img_shape (optional)
        cam0_key:          obs["splat"] key for view 0 — front camera (default: cam0)
        cam1_key:          obs["splat"] key for view 1 — left  camera (default: cam1)
    """

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        host = args.host if args.host is not None else "localhost"
        port = args.port if args.port is not None else 5557
        open_loop_horizon = args.open_loop_horizon if args.open_loop_horizon is not None else 1
        self.open_loop_horizon = open_loop_horizon

        # Read image shape from checkpoint if available
        ckpt_path = args.policy_path if isinstance(args.policy_path, str) else (args.policy_path[0] if args.policy_path else None)
        shape = _read_img_shape_from_ckpt(ckpt_path) if ckpt_path else None
        self.image_h = shape[0] if shape else _DEFAULT_IMAGE_H
        self.image_w = shape[1] if shape else _DEFAULT_IMAGE_W

        # Camera keys in obs["splat"] → AMPLIFY view order [front, left]
        self.cam_keys = [
            getattr(args, "cam0_key", "cam0"),  # front view
            getattr(args, "cam1_key", "cam1"),  # left view
        ]

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to AMPLIFY server at {host}:{port}")
        print(f"Camera mapping: {self.cam_keys[0]} → front, {self.cam_keys[1]} → left")
        print(f"Image size: {self.image_h}×{self.image_w}")

        # curobo IK solver for EEF pose → joint positions
        from polaris.utils_.planner_utils import setup_curobo
        self.motion_gen = setup_curobo()

        self.action_chunk = None
        self.actions_from_chunk_completed = 0
        self.last_response = {}

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.action_chunk = None
        self.actions_from_chunk_completed = 0
        self.last_response = {}
        self.socket.send(pickle.dumps({"reset": True}))
        self.socket.recv()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        image, proprio = self._extract_observation(obs)

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            request = {"image": image, "proprio": proprio}
            self.socket.send(pickle.dumps(request))
            self.last_response = pickle.loads(self.socket.recv())
            self.action_chunk = self.last_response["action_chunk"]  # (action_horizon, 10)
            self.actions_from_chunk_completed = 0

        action10 = self.action_chunk[self.actions_from_chunk_completed]  # (10,)
        self.actions_from_chunk_completed += 1

        env_action = self._eef_to_joint_action(action10, obs)  # (8,)

        viz = None
        if return_viz:
            if "vis_frame" in self.last_response:
                # Server returned track-overlaid image: (H, v*W, 3) uint8
                viz = cv2.resize(self.last_response["vis_frame"], (VIZ_W, VIZ_H))
            else:
                viz = cv2.resize(obs["splat"]["cam1"], (426, VIZ_H))

        return env_action, viz

    def _eef_to_joint_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
        """
        Convert EEF action (10,) = pos(3) + rot6d(6) + gripper(1)
        to joint action (8,) = joint_pos(7) + gripper(1) via curobo IK.
        Gripper: 0=open, 1=closed.
        """
        from curobo.types.math import Pose

        pos       = action[:3]
        rot6d     = action[3:9]
        gripper   = action[9]

        quat_wxyz = rot6d_to_quat_wxyz(rot6d)                                 # (4,)

        ee_pos  = torch.from_numpy(pos).float().unsqueeze(0).to(DEVICE)       # (1, 3)
        ee_quat = torch.from_numpy(quat_wxyz).float().unsqueeze(0).to(DEVICE) # (1, 4) wxyz

        goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)

        if ik_result.success.item():
            joint_pos = ik_result.solution.squeeze(0)[0, :7].cpu().numpy()  # (7,)
        else:
            print("[AMPLIFYFullResClient] IK failed, holding current joints")
            joint_pos = obs["policy"]["arm_joint_pos"][0].cpu().numpy()     # (7,)

        gripper_bin = 1.0 if gripper > 0.5 else 0.0

        return np.concatenate([joint_pos, [gripper_bin]]).astype(np.float32)  # (8,)

    def _extract_observation(self, obs: dict):
        """
        Returns:
            image:   (2, H, W, 3) float32 [0, 1]  — H×W from checkpoint img_shape
            proprio: (10,) float32 = pos(3) + rot6d(6) + gripper(1)
        """
        images = []
        for key in self.cam_keys:
            raw = obs["splat"][key]  # (H, W, 3) uint8
            if raw.shape[0] != self.image_h or raw.shape[1] != self.image_w:
                raw = cv2.resize(raw, (self.image_w, self.image_h))
            images.append(raw)
        image = np.stack(images, axis=0).astype(np.float32) / 255.0

        # ee_pose: pos(3) + quat_wxyz(4) + gripper(1) from simulator
        ee_pose = obs["policy"]["ee_pose"][0].cpu().numpy().astype(np.float32)  # (8,)
        pos     = ee_pose[:3]
        rot6d   = quat_wxyz_to_rot6d(ee_pose[3:7])  # wxyz → xyzw → rot6d (6,)
        gripper = ee_pose[7:8]
        proprio = np.concatenate([pos, rot6d, gripper])  # (10,)

        return image, proprio
