"""
Ghost WebSocket client.

"Ghost" means we use the ground-truth goal_gripper_pcd (oracle) from the
observation instead of running the HighLevelWrapper goal predictor.

Flow:
    obs["policy"]["goal_gripper_pcd"]  (world frame, 4×3)
        ↓  extrinsics  →  camera frame
        ↓  intrinsics  →  2-D pixel coords
        ↓  heatmap     →  (H, W, 3) uint8
        ↓  WebSocket   →  lerobot_diffusion_server (ghost variant)
        ↓  EE action   ←  server
        ↓  CuRobo IK   →  joint-space action (1, 8)

Matching server: src/polaris/server/ghost_server.py
"""

import asyncio
from typing import Optional

import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np
import torch
import websockets
from scipy.spatial.transform import Rotation as R

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs


# ---------------------------------------------------------------------------
# Heatmap utilities (mirrors lerobot/scripts/dataset_utils.py and
# lerobot/common/policies/high_level/high_level_wrapper.py::project)
# ---------------------------------------------------------------------------

def get_heatmap_viz(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a distance heatmap on an RGB image (goal center appears red).

    Args:
        rgb_image: (H, W, 3) uint8 RGB
        heatmap:   (H, W) uint8 distance heatmap (higher = farther from goal)
        alpha:     heatmap blend weight

    Returns:
        (H, W, 3) uint8 blended image
    """
    import cv2
    gray = np.asarray(heatmap, dtype=np.uint8)
    gray_inv = 255 - gray                                     # invert: goal = bright
    colored_bgr = cv2.applyColorMap(gray_inv, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blended = rgb_image.astype(np.float32) * (1 - alpha) + colored_rgb.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)

def _generate_heatmap(points_2d: np.ndarray, img_shape: tuple) -> np.ndarray:
    """
    3-channel distance heatmap from N×2 projected pixel coords.
    Channels 0/1/2 encode distance to gripper points 0/1/2 (tip/left/right).
    Returns (H, W, 3) uint8.
    """
    height, width = img_shape[:2]
    max_dist = np.sqrt(width ** 2 + height ** 2)

    clipped = np.clip(points_2d[:3], [0, 0], [width - 1, height - 1]).astype(int)

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_grid = np.stack([x_coords, y_coords], axis=-1)  # (H, W, 2)

    heatmap = np.empty((height, width, 3), dtype=np.float32)
    for i in range(3):
        heatmap[:, :, i] = np.linalg.norm(pixel_grid - clipped[i], axis=-1)

    heatmap = np.sqrt(heatmap / max_dist) * 255.0
    return np.clip(heatmap, 0, 255).astype(np.uint8)


def _project_goal_pcd(
    goal_pcd_world: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    img_shape: tuple,
) -> np.ndarray:
    """
    Project goal gripper PCD (world frame) to a goal-conditioning heatmap.

    Args:
        goal_pcd_world: (4, 3) world-frame gripper points
        K:              (3, 3) camera intrinsics
        cam_to_world:   (4, 4) camera-to-world extrinsic
        img_shape:      (H, W) output image size

    Returns:
        (H, W, 3) uint8 heatmap
    """
    world_to_cam = np.linalg.inv(cam_to_world)

    # world → camera frame
    pts_cam = (world_to_cam[:3, :3] @ goal_pcd_world.T).T + world_to_cam[:3, 3]

    # camera → image  (pinhole)
    pts_2d_hom = (K @ pts_cam.T).T          # (4, 3)
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]  # (4, 2)

    return _generate_heatmap(pts_2d, img_shape)


# ---------------------------------------------------------------------------
# Helper: normalize quaternion (wxyz convention)
# ---------------------------------------------------------------------------

def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < eps or not np.isfinite(norm):
        q = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        q = q / norm
    if q[np.argmax(np.abs(q))] < 0:
        q = -q
    return q


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@InferenceClient.register(client_name="ghost")
class GhostClient(InferenceClient):
    """
    Goal-conditioned LeRobot diffusion policy client with oracle goal projection
    """

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        host = getattr(args, "host", None) or "localhost"
        port = getattr(args, "port", None) or 8768

        self.uri = f"ws://{host}:{port}"
        print(f"[GhostClient] will connect → {self.uri}")

        self.device = getattr(args, "device", "cuda")
        self._rerender = True
        self._latest_goal_heatmap_front: Optional[np.ndarray] = None
        self._latest_goal_heatmap_left:  Optional[np.ndarray] = None

        # Camera calibration for oracle goal projection
        # intrinsics = getattr(args, "intrinsics_txt", None)
        # extrinsics = getattr(args, "extrinsics_txt", None)
        # self.K: Optional[np.ndarray] = np.loadtxt(intrinsics) if intrinsics else None
        # self.cam_to_world: Optional[np.ndarray] = np.loadtxt(extrinsics) if extrinsics else None
        # if self.K is None or self.cam_to_world is None:
        #     print("[GhostClient] WARNING: intrinsics/extrinsics not set; "
        #           "goal heatmap will be zeros.")

        self.img_w: int = getattr(args, "img_width",  1280)
        self.img_h: int = getattr(args, "img_height",  720)

        # CuRobo IK for EE → joint conversion
        from polaris.utils_.planner_utils import setup_curobo
        self.motion_gen = setup_curobo()

    # ------------------------------------------------------------------
    # InferenceClient protocol
    # ------------------------------------------------------------------

    @property
    def rerender(self) -> bool:
        return self._rerender

    def reset(self) -> None:
        self._rerender = True
        self._latest_goal_heatmap_front = None
        self._latest_goal_heatmap_left  = None
        asyncio.run(self._send({"reset": True}))

    def infer(
        self,
        obs: dict,
        instruction: str = "",
        return_viz: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        curr = self._extract_observation(obs)

        response = asyncio.run(self._send({
            "cam0_rgb":    curr["cam0_rgb"],
            "cam0_depth":  curr["cam0_depth"],
            "cam1_rgb":    curr["cam1_rgb"],
            "cam1_depth":  curr["cam1_depth"],
            "wrist_cam":   curr["wrist_cam"],
            "ee_pose":     curr["ee_pose"],
            "gripper_pcd": curr["gripper_pcd"],
            "state":       curr["state"],
            "instruction": instruction,
        }))

        self._rerender = response.get("rerender", True)

        if "action_ee" in response:
            action_ee = np.asarray(response["action_ee"], dtype=np.float32)
            action_joint = self._ee_to_joint(action_ee)
        else:
            action_joint = np.asarray(response["action_joint"], dtype=np.float32).reshape(1, 8)

        # Update cached goal heatmaps when server sends fresh ones (numpy arrays via msgpack_numpy)
        front = response.get("goal_heatmap_front")
        left  = response.get("goal_heatmap_left")
        if front is not None:
            self._latest_goal_heatmap_front = np.asarray(front, dtype=np.uint8)
        if left is not None:
            self._latest_goal_heatmap_left = np.asarray(left, dtype=np.uint8)

        viz = self._make_viz(curr) if return_viz else None
        return action_joint, viz

    def shutdown(self) -> None:
        try:
            asyncio.run(self._send({"shutdown": True}))
            print("[GhostClient] shutdown sent.")
        except Exception as exc:
            print(f"[GhostClient] server already closed: {exc}")

    async def _send(self, message: dict) -> dict:
        async with websockets.connect(self.uri, max_size=100 * 1024 * 1024, ping_interval=None) as ws:
            await ws.send(msgpack.packb(message, use_bin_type=True))
            response = await asyncio.wait_for(ws.recv(), timeout=120.0)
            return msgpack.unpackb(response, raw=False)

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------

    def _extract_observation(self, obs_dict: dict) -> dict:
        cam0      = obs_dict["splat"]["cam0"]       # (H, W, 3) uint8 RGB  — front / azure-kinect
        cam1      = obs_dict["splat"]["cam1"]       # (H, W, 3) uint8 RGB  — left  / azure-kinect
        cam0_depth = obs_dict["splat"]["cam0_depth"] # (H, W) float32 depth in meters
        cam1_depth = obs_dict["splat"]["cam1_depth"] # (H, W) float32 depth in meters
        wrist_cam = obs_dict["splat"]["wrist_cam"]  # (H, W, 3) uint8 RGB

        # Convert meters → mm and clip inf/nan (background pixels) to 0, matching
        # real-camera training data where invalid pixels are 0.
        cam0_depth = np.clip(cam0_depth * 1000, 0, 65535).astype(np.uint16)
        cam1_depth = np.clip(cam1_depth * 1000, 0, 65535).astype(np.uint16)

        robot_state = obs_dict["policy"]

        ee_pose = robot_state["ee_pose"]
        state = torch.cat([robot_state["arm_joint_pos"], robot_state["gripper_pos"]], dim=1).squeeze(0).cpu().numpy()
        if isinstance(ee_pose, torch.Tensor):
            ee_pose = ee_pose.detach().cpu().numpy()
        ee_pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)[:8]  # xyz+quat_wxyz+gripper

        gripper_pcd = robot_state["gripper_pcd"][:, [1, 2, 0, 3]] # reorder !!!!
        if isinstance(gripper_pcd, torch.Tensor):
            gripper_pcd = gripper_pcd.detach().cpu().numpy()
        gripper_pcd = np.asarray(gripper_pcd, dtype=np.float32).reshape(4, 3)

        return {
            "cam0_rgb":             cam0,
            "cam1_rgb":             cam1,
            "cam0_depth":       cam0_depth,
            "cam1_depth":       cam1_depth,
            "wrist_cam":        wrist_cam,
            "ee_pose":          ee_pose,
            "gripper_pcd":      gripper_pcd,
            "state" :              state,
        }

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def _ee_to_joint(self, action_ee: np.ndarray) -> np.ndarray:
        """Convert EE pose → joint positions (1, 8).

        Accepts two formats:
          - 10-dim: rot6d(6) + xyz(3) + gripper(1)   [right_eef policy output]
          - 8-dim:  xyz(3) + quat_wxyz(4) + gripper(1)
        """
        from curobo.types.math import Pose

        if action_ee.shape[0] == 10:
            #  xyz(3) + rot6d(6) + gripper(1)
            pos_np  = action_ee[:3]
            import pytorch3d.transforms as pt3d
            rot6d_t = torch.tensor(action_ee[3:9], dtype=torch.float32).unsqueeze(0)
            rot_mat = pt3d.rotation_6d_to_matrix(rot6d_t).squeeze(0).numpy()  # (3,3)
            q_xyzw  = R.from_matrix(rot_mat).as_quat()
            q_wxyz  = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
            
            gripper = float(action_ee[9])
        else:
            pos_np  = action_ee[:3]
            q_wxyz  = action_ee[3:7].astype(np.float64)
            gripper = float(action_ee[7])

        pos  = torch.tensor(pos_np, dtype=torch.float32, device=self.device)
        quat = torch.tensor(
            _normalize_quat_wxyz(q_wxyz),
            dtype=torch.float32,
            device=self.device,
        )

        goal_pose = Pose(position=pos, quaternion=quat)
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
        joints = ik_result.solution.squeeze(0)[:, :8]   # (1, 8)
        joints[:, -1] = gripper
        return joints.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _make_viz(self, curr: dict) -> np.ndarray:
        import cv2
        ext0  = curr["cam0_rgb"]
        ext1  = curr["cam1_rgb"]
        if self._latest_goal_heatmap_front is not None:
            hmap = self._latest_goal_heatmap_front
            ext0 = get_heatmap_viz(ext0, hmap[:, :, 0])
        if self._latest_goal_heatmap_left is not None:
            hmap = self._latest_goal_heatmap_left
            ext1 = get_heatmap_viz(ext1, hmap[:, :, 0])
        return np.concatenate([ext0, ext1], axis=1)
