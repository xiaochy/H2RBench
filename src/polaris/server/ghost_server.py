"""
Ghost policy inference server for goal-conditioned LeRobot Diffusion policy.

Pairs with GhostLeRobotClient (src/polaris/client/ghost_lerobot_client.py).
The client sends raw observations; this server does ALL preprocessing to match
exactly what the policy saw during training.

Obs keys built here:
  observation.images.cam_azure_kinect_front.color       ← cam0
  observation.images.cam_azure_kinect_left.color        ← cam1
  observation.images.cam_wrist                          ← wrist_cam
  observation.images.cam_azure_kinect_front.goal_gripper_proj  ← goal_pcd projected
  observation.images.cam_azure_kinect_left.goal_gripper_proj   ← goal_pcd projected
  observation.state                                     ← ee_pose (8-dim raw)
  observation.right_eef_pose                            ← quat→rot6d, 10-dim
  observation.points.gripper_pcds                       ← gripper_pcd (4,3)
  observation.points.goal_gripper_pcds                  ← goal_gripper_pcd (4,3)
  observation.cam_azure_kinect_front.intrinsics         ← K_front (3,3)
  observation.cam_azure_kinect_front.extrinsics         ← world_to_cam_front (4,4)
  observation.cam_azure_kinect_left.intrinsics          ← K_left (3,3)
  observation.cam_azure_kinect_left.extrinsics          ← world_to_cam_left (4,4)

Usage:
    conda activate robodiff
    python src/polaris/server/ghost_server.py \
        --policy_path /home/haotian/lerobot/outputs/train/pick_place_red_mug_r40h100/checkpoints/last/pretrained_model \
        --open_loop_horizon 8 \
        [--port 5555]
"""

import argparse
import asyncio
import os
import sys
from collections import deque
import pathlib

# Use the polaris-internal lerobot (matches training code)
ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "policy" / "lerobot")
sys.path.insert(0, ROOT_DIR)

import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import websockets

from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.pretrained import PreTrainedConfig
import lerobot


N_OBS_STEPS = 2
CALIB_DIR = os.path.join(ROOT_DIR, "lerobot", "scripts", "droid_calibration")


# ---------------------------------------------------------------------------
# Camera calibration (loaded once at startup)
# ---------------------------------------------------------------------------

def load_calibration():
    K_front    = np.loadtxt(os.path.join(CALIB_DIR, "intrinsics_front.txt"))           # (3,3)
    K_left     = np.loadtxt(os.path.join(CALIB_DIR, "intrinsics_left.txt"))            # (3,3)
    T_front    = np.loadtxt(os.path.join(CALIB_DIR, "T_world_from_camera_front.txt"))  # (4,4) cam→world
    T_left     = np.loadtxt(os.path.join(CALIB_DIR, "T_world_from_camera_left.txt"))   # (4,4) cam→world
    return K_front, K_left, T_front, T_left


# ---------------------------------------------------------------------------
# Heatmap generation (mirrors high_level_wrapper.py::project)
# ---------------------------------------------------------------------------

def _project_goal_pcd_to_heatmap(
    goal_pcd_world: np.ndarray,   # (4, 3) world-frame
    K: np.ndarray,                # (3, 3) intrinsics
    cam_to_world: np.ndarray,     # (4, 4)
    img_shape: tuple,             # (H, W)
) -> np.ndarray:                  # (H, W, 3) uint8
    world_to_cam = np.linalg.inv(cam_to_world)
    pts_cam = (world_to_cam[:3, :3] @ goal_pcd_world.T).T + world_to_cam[:3, 3]
    pts_2d_hom = (K @ pts_cam.T).T
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]

    H, W = img_shape
    max_dist = np.sqrt(W**2 + H**2)
    clipped = np.clip(pts_2d[:3], [0, 0], [W - 1, H - 1]).astype(int)
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    pixel_grid = np.stack([x_coords, y_coords], axis=-1)

    heatmap = np.empty((H, W, 3), dtype=np.float32)
    for i in range(3):
        heatmap[:, :, i] = np.linalg.norm(pixel_grid - clipped[i], axis=-1)
    heatmap = np.sqrt(heatmap / max_dist) * 255.0
    return np.clip(heatmap, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ee_pose → right_eef_pose  (rot6d + xyz + gripper, 10-dim)
# ---------------------------------------------------------------------------

def ee_pose_to_right_eef_pose(ee_pose: np.ndarray) -> np.ndarray:
    """
    ee_pose: (8,) [xyz(3) + quat_wxyz(4) + gripper(1)]
    returns: (10,) [rot6d(6) + xyz(3) + gripper(1)]
    rot6d = first two rows of rotation matrix, flattened  (pytorch3d convention)
    """
    xyz     = ee_pose[:3]
    q_wxyz  = ee_pose[3:7]
    gripper = ee_pose[7:8]

    q_xyzw  = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    rot_mat = R.from_quat(q_xyzw).as_matrix()   # (3, 3)
    rot_6d  = rot_mat[:2, :].reshape(6)          # first two rows → 6-dim

    return np.concatenate([xyz, rot_6d, gripper]).astype(np.float32)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 → (1, C, H, W) float32 CUDA [0,1]."""
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to("cuda")


def arr_to_tensor(arr: np.ndarray, shape=None) -> torch.Tensor:
    """np array → (1, *shape) float32 CUDA."""
    t = torch.from_numpy(arr.astype(np.float32))
    if shape is not None:
        t = t.reshape(shape)
    return t.unsqueeze(0).to("cuda")


# ---------------------------------------------------------------------------
# Build full lerobot observation dict
# ---------------------------------------------------------------------------

def build_obs(request: dict, calib: tuple) -> dict:
    K_front, K_left, T_front, T_left = calib

    cam0_rgb      = request["cam0_rgb"]           # (H, W, 3) uint8
    cam1_rgb      = request["cam1_rgb"]           # (H, W, 3) uint8
    cam0_depth    = request["cam0_depth"]         # (H, W) uint16 depth in mm
    cam1_depth    = request["cam1_depth"]         # (H, W) uint16 depth in mm
    
    wrist_cam = request["wrist_cam"]      # (H, W, 3) uint8
    ee_pose   = request["ee_pose"]        # (8,) float32
    gripper_pcd      = request["gripper_pcd"]       # (4, 3) float32

    state = request["state"]

    H, W = cam0_rgb.shape[:2]
    img_shape = (H, W)

    # EEF pose in rot6d format
    right_eef_pose = ee_pose_to_right_eef_pose(ee_pose)

    # world_to_cam extrinsics for the obs dict
    world_to_cam_front = np.linalg.inv(T_front)  # (4,4)
    world_to_cam_left  = np.linalg.inv(T_left)   # (4,4)

    obs = {
        # ---- images ----
        "observation.images.cam_azure_kinect_front.color":            img_to_tensor(cam0_rgb),
        "observation.images.cam_azure_kinect_left.color":             img_to_tensor(cam1_rgb),
        "observation.images.cam_azure_kinect_front.transformed_depth": arr_to_tensor(cam0_depth/1000.0),  # convert mm → m
        "observation.images.cam_azure_kinect_left.transformed_depth":  arr_to_tensor(cam1_depth/1000.0),  # convert mm → m
        "observation.images.cam_wrist":                               img_to_tensor(wrist_cam),

        # ---- state ----
        "observation.state":           arr_to_tensor(state),                   # (1, 8)
        "observation.right_eef_pose":  arr_to_tensor(right_eef_pose),            # (1, 10)
        # ---- point clouds ----
        "observation.points.gripper_pcds":      arr_to_tensor(gripper_pcd,      (4, 3)),  # (1, 4, 3)
        # ---- camera calibration tensors ----
        "observation.cam_azure_kinect_front.intrinsics": arr_to_tensor(K_front,           (3, 3)),  # (1, 3, 3)
        "observation.cam_azure_kinect_front.extrinsics": arr_to_tensor(world_to_cam_front, (4, 4)), # (1, 4, 4)
        "observation.cam_azure_kinect_left.intrinsics":  arr_to_tensor(K_left,            (3, 3)),  # (1, 3, 3)
        "observation.cam_azure_kinect_left.extrinsics":  arr_to_tensor(world_to_cam_left,  (4, 4)), # (1, 4, 4)
        # ---- task ----
        "task": [request["instruction"]],
    }

    return obs


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(policy_path: str):
    os.chdir(ROOT_DIR)  # relative calib paths resolve from here

    policy_config: PreTrainedConfig = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path

    lerobot_dir = os.path.dirname(os.path.dirname(lerobot.__file__))
    if hasattr(policy_config, "calibration_json") and policy_config.calibration_json:
        policy_config.calibration_json = os.path.join(lerobot_dir, policy_config.calibration_json)

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(config=policy_config, pretrained_name_or_path=policy_path)
    policy.eval()
    policy.to("cuda")
    return policy


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path",       type=str, required=True)
    parser.add_argument("--open_loop_horizon", type=int, default=8)
    parser.add_argument("--port",              type=int, default=8768)
    args = parser.parse_args()

    print(f"Loading calibration from {CALIB_DIR} ...")
    calib = load_calibration()
    print("Calibration loaded.")

    print(f"Loading policy from {args.policy_path} ...")
    policy = load_policy(args.policy_path)
    print(f"Policy loaded.  open_loop_horizon={args.open_loop_horizon}")

    steps_since_infer = 0

    async def handle(websocket):
        nonlocal steps_since_infer
        async for raw in websocket:
            request = msgpack.unpackb(raw, raw=False)

            # ---- reset ----------------------------------------------------
            if request.get("reset"):
                steps_since_infer = 0
                policy.reset()
                torch.cuda.empty_cache()
                await websocket.send(msgpack.packb({"status": "ok"}, use_bin_type=True))
                continue

            # ---- shutdown -------------------------------------------------
            if request.get("shutdown"):
                await websocket.send(msgpack.packb({"status": "shutdown"}, use_bin_type=True))
                print("Shutdown requested — exiting.")
                return

            # ---- inference step -------------------------------------------
            try:
                lerobot_obs = build_obs(request, calib)

                with torch.inference_mode():
                    enable_goal_condition = (
                        hasattr(policy.config, "enable_goal_conditioning")
                        and policy.config.enable_goal_conditioning
                        and (policy._queues is None or len(policy._queues[policy.act_key]) == 0)
                    )
                    latest_heatmap_front = None
                    latest_heatmap_left  = None
                    if enable_goal_condition:
                        cam_name_front = policy.high_level.camera_names[0]
                        cam_name_left  = policy.high_level.camera_names[1]

                        camera_obs = {
                            cam_name_front: {"rgb": request["cam0_rgb"], "depth": request["cam0_depth"]},
                            cam_name_left:  {"rgb": request["cam1_rgb"], "depth": request["cam1_depth"]},
                        }
                        robot_kwargs = {
                            "observation.state": lerobot_obs["observation.state"],
                            "gripper_pcd":       request["gripper_pcd"],
                        }
                        gripper_projs = policy.high_level.predict_and_project(
                            request["instruction"],
                            camera_obs,
                            robot_type=policy.config.robot_type,
                            robot_kwargs=robot_kwargs,
                        )
                        for cam_name, proj in gripper_projs.items():
                            policy.latest_gripper_proj[cam_name] = img_to_tensor(proj)
                        latest_heatmap_front = gripper_projs.get(cam_name_front)
                        latest_heatmap_left  = gripper_projs.get(cam_name_left)

                    for cam_name, proj_tensor in policy.latest_gripper_proj.items():
                        lerobot_obs[f"observation.images.{cam_name}.goal_gripper_proj"] = proj_tensor

                    action_joint, action_eef = policy.select_action(lerobot_obs)

                steps_since_infer += 1
                if steps_since_infer >= args.open_loop_horizon:
                    steps_since_infer = 0
                rerender = (steps_since_infer == 0)

                resp = {"rerender": rerender,
                        "goal_heatmap_front": latest_heatmap_front,
                        "goal_heatmap_left":  latest_heatmap_left}
                if policy.config.action_space in ("right_eef", "right_eef_relative"):
                    action = action_eef.squeeze(0).cpu().numpy()
                    action = np.concatenate([
                        action[:-1],
                        np.ones((1,)) if action[-1] > 0.5 else np.zeros((1,))
                    ])
                    resp["action_ee"] = action
                elif policy.config.action_space == "joint":
                    action = action_joint.squeeze(0).cpu().numpy()
                    action = np.concatenate([
                        action[:-1],
                        np.ones((1,)) if action[-1] > 0.5 else np.zeros((1,))
                    ])
                    resp["action_joint"] = action
                else:
                    raise ValueError(f"Unsupported action space: {policy.config.action_space}")

                print("action_space", policy.config.action_space, "action", action)

            except Exception:
                import traceback
                traceback.print_exc()
                resp = {"error": "inference failed"}

            await websocket.send(msgpack.packb(resp, use_bin_type=True))

    async def serve():
        async with websockets.serve(handle, "0.0.0.0", args.port, max_size=100 * 1024 * 1024,
                                    ping_interval=None):
            print(f"Ghost server listening on port {args.port}")
            await asyncio.Future()  # run forever

    asyncio.run(serve())


if __name__ == "__main__":
    main()
