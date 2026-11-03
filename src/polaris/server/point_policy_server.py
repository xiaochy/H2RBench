"""
Server script for PointPolicy — runs in point-policy env.

Run with:
    python point_policy_server.py \ 
    --bc_weight /home/haotian/Point-Policy/point_policy/exp_local/2026.04.03/point_policy/deterministic/143730_hidden_dim_256/snapshot/100000.pt \
    --port 8765 \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" "suite.use_robot_points=true" "suite.use_object_points=true" "experiment=eval_point_policy" "suite/task/franka_env=pick_place_red_mug"
"""
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "policy" / "Point-Policy" / "point_policy")
sys.path.insert(0, ROOT_DIR)

import asyncio
import dataclasses
import json
import logging
import socket
import cv2
from pathlib import Path
from typing import List

import msgpack
import msgpack_numpy as m
m.patch()

import hydra
import numpy as np
import torch
import tyro
import websockets
from omegaconf import OmegaConf

import utils
from eval_point_track import Workspace
import yaml
from point_utils.points_class import PointsClass
from robot_utils.franka.utils import pixel2d_to_3d_torch, triangulate_points

from robot_utils.franka.utils import ee_pose_to_robot_points, project_points, robot_points_to_ee_pose_with_gripper, camera2pixelkey, camera_indices
from scipy.spatial.transform import Rotation as R
from robot_utils.franka.utils import rigid_transform_3D
from robot_utils.franka.gripper_points import Tshift

@dataclasses.dataclass
class Args:
    bc_weight: str
    port: int = 8765
    config_path: str = ""
    config_name: str = "config"
    overrides: List[str] = dataclasses.field(default_factory=list)


def numpy_from_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = numpy_from_dict(v)
        elif isinstance(v, list):
            out[k] = np.array(v)
        else:
            out[k] = v
    return out


def numpy_to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    if isinstance(v, dict):
        return {k: numpy_to_list(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [numpy_to_list(i) for i in v]
    return v

def draw_tracks(img, obj_2d, rob_2d):
        img = img.copy()

        for i, pt in enumerate(obj_2d):
            try:
                if not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                    continue
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
                cv2.putText(img, f"obj{i}", (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except (ValueError, OverflowError, cv2.error):
                continue
        for i, pt in enumerate(rob_2d):
            try:
                if not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                    continue
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img, f"rob{i}", (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except (ValueError, OverflowError, cv2.error):
                continue
        return img

class PointPolicyServer:
    def __init__(self, args: Args):
        # mirror exactly what eval.py does
        config_dir = str(pathlib.Path(ROOT_DIR) / (args.config_path or "cfgs"))
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(
                config_name=args.config_name,
                overrides=args.overrides + [f"bc_weight={args.bc_weight}"],
            )

        logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        self.workspace = Workspace(cfg)

        bc_snapshot = Path(args.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        logging.info("Loading bc weight: %s", bc_snapshot)
        self.workspace.load_snapshot({"bc": bc_snapshot})
        self.workspace.agent.train(False)

        CALIB_PATH = Path(cfg.root_dir) / "calib" / "calib.npy"
        self.calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

        self.step = 0
        logging.info("PointPolicy loaded.")
        
        with open(str(pathlib.Path(ROOT_DIR) / "cfgs" / "suite" / "points_cfg.yaml")) as stream:
            try:
                points_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        root_dir = points_cfg["root_dir"]
        points_cfg["dift_path"]            = f"{root_dir}/{points_cfg['dift_path']}"
        points_cfg["cotracker_checkpoint"] = f"{root_dir}/{points_cfg['cotracker_checkpoint']}"
        points_cfg["pixel_keys"]           = [camera2pixelkey[f"cam_{i}"] for i in camera_indices]
        points_cfg["object_labels"]        = cfg.suite.task.object_labels
        points_cfg["task_name"]            = cfg.suite.task.franka_env.task_name
        self.points_class = PointsClass(**points_cfg)

        self.use_robot_points = cfg.suite.use_robot_points
        self.num_robot_points = cfg.suite.num_robot_points
        self.num_object_points = cfg.suite.num_object_points

        self.prev_gripper_state = -1

    def reset(self):
        self.workspace.agent.buffer_reset()
        self.step = 0
        self.prev_gripper_state = -1

        self.is_first_frame = True
        self._track_pts = {}
        self.points_class.reset_episode()

        # self.process_point(obs)

    @torch.no_grad()
    def infer(self, obs: dict, return_viz: bool = False) -> np.ndarray:
        frame = None
        with utils.eval_mode(self.workspace.agent):
            action = self.workspace.agent.act(
                obs,
                self.workspace.expert_replay_loader.dataset.stats,
                self.step,
                self.step,
                eval_mode=True,
            )
        self.step += 1

        if return_viz:
            img1 = draw_tracks(obs["pixels1"].astype(np.uint8), obs["object_tracks_pixels1"], obs["robot_tracks_pixels1"])
            img2 = draw_tracks(obs["pixels2"].astype(np.uint8), obs["object_tracks_pixels2"], obs["robot_tracks_pixels2"])
            frame = np.concatenate([img1, img2], axis=1)

        return action, frame

    @torch.no_grad()
    def process_point(self, obs: dict) -> dict:
        """
        Process observation to add point tracks.
        First frame: init tracking (expensive - DIFT + CoTracker)
        Subsequent frames: update tracking (fast - CoTracker only)
        """

        camera_indices = [1, 2]
        camera2pixelkey = {
            "cam_1": "pixels1",
            "cam_2": "pixels2",
        }
        pixelkey2camera = {
            "pixels1": "cam_1",
            "pixels2": "cam_2",
        }
        
        pixel_keys = ["pixels1", "pixels2"]

        # --- Robot points (every frame) ---
        gripper_pcd = obs["gripper_pcd"]
        states_ee = obs["states_ee"]
        robot_points_3d, gripper_states = ee_pose_to_robot_points(gripper_pcd, states_ee)
        robot_points_3d = robot_points_3d[0]  # (N, 3)

        # Project robot points to 2D for each camera
        robot_points_2d = {}
        for cam_idx in camera_indices:
            camera_name = f"cam_{cam_idx}"
            pixel_key = camera2pixelkey[camera_name]
            robot_2d, _ = project_points(
                self.calibration_data,
                robot_points_3d[None],
                np.zeros((1, self.num_object_points, 3)),
                camera_name
            )
            robot_points_2d[pixel_key] = robot_2d[0]  # (N, 2)

        object_points_2d = {}
        if self.is_first_frame:
            for pixel_key in pixel_keys:
                points = []

                # Robot points
                if self.use_robot_points:
                    points.append(robot_points_2d[pixel_key])

                # Object points - DIFT + CoTracker init
                if self.num_object_points > 0:
                    frame = obs[pixel_key]  # RGB
                    self.points_class.add_to_image_list(frame, pixel_key)
                    
                    for object_label in ["objects"]:
                        self.points_class.find_semantic_similar_points(pixel_key, object_label)
                    
                    self.points_class.track_points(pixel_key, is_first_step=True)
                    self.points_class.track_points(pixel_key)
                    object_pts = self.points_class.get_points_on_image(pixel_key).numpy()[0]

                    object_points_2d[pixel_key] = object_pts

                    if object_pts.shape[0] != self.num_object_points or object_pts is None:
                        print(f"WARNING: [{pixel_key}] no object points, expected {self.num_object_points}")
                    points.append(object_pts)

                self._track_pts[pixel_key] = np.concatenate(points, axis=0)

            self.is_first_frame = False

        else:
            for pixel_key in pixel_keys:
                current_track = robot_points_2d[pixel_key]

                if self.num_object_points > 0:
                    frame = obs[pixel_key]
                    self.points_class.add_to_image_list(frame, pixel_key)
                    self.points_class.track_points(pixel_key)
                    object_pts = self.points_class.get_points_on_image(pixel_key).numpy()[0]
                    object_points_2d[pixel_key] = object_pts
                    current_track = np.concatenate([current_track, object_pts], axis=0)

                self._track_pts[pixel_key] = current_track

        # --- Triangulate 2D → 3D ---
        P_list, pts_list = [], []
        for cam_idx in camera_indices:
            camera_name = f"cam_{cam_idx}"
            pixel_key = camera2pixelkey[camera_name]
            
            extr = self.calibration_data[camera_name]["ext"]
            intr = self.calibration_data[camera_name]["int"]
            P_list.append(np.concatenate([intr, np.zeros((3, 1))], axis=1) @ extr)
            pts_list.append(self._track_pts[pixel_key])

        pts3d = triangulate_points(P_list, pts_list)[:, :3]

        # Replace robot points with ground truth 3D
        pts3d[:self.num_robot_points] = robot_points_3d

        # --- Store in obs ---
        for pixel_key in pixel_keys:
            obs[f"point_tracks_{pixel_key}"] = pts3d
            obs[f"robot_tracks_{pixel_key}"] = robot_points_2d[pixel_key]
            obs[f"object_tracks_{pixel_key}"] = object_points_2d[pixel_key]

        obs["gripper_states"] = gripper_states
        features = states_ee.squeeze().copy()
        features[-1] = float(gripper_states)
        obs["features"] = features

        return obs


    def point2action(self, action):
        """
        Convert predicted future point tracks to robot action.
        
        action dict contains:
            future_tracks_pixels1: (N, 3) - predicted 3D points for camera 1
            future_tracks_pixels2: (N, 3) - predicted 3D points for camera 2
            gripper: (1,) - gripper action
        """
        pixel_keys = ["pixels1", "pixels2"]
        
        points = []
        for pixel_key in pixel_keys:
            # Get robot points only (first num_robot_points)
            future_tracks = action[f"future_tracks_{pixel_key}"][:self.num_robot_points, :3]
            points.append(future_tracks)
        
        # Average 3D points from both cameras
        points3d = np.mean(points, axis=0)  # (num_robot_points, 3)
        
        gripper_state = self.compute_gripper(action)
        robot_action = robot_points_to_ee_pose_with_gripper(points3d, gripper_state)
        
        return robot_action
    
    def compute_gripper(self, action):
        gripper_state = action["gripper"][:1]

        if self.prev_gripper_state == -1 and gripper_state > -0.3:
            gripper_state = 1
        elif self.prev_gripper_state == 1 and gripper_state < 0.6:
            gripper_state = -1
        else:
            gripper_state = self.prev_gripper_state
        self.prev_gripper_state = gripper_state

        gripper_state = gripper_state
        return gripper_state

def main(args: Args):
    server = PointPolicyServer(args)
    shutdown_event = asyncio.Event() 

    async def handle(websocket):
        async for message in websocket:
            data = msgpack.unpackb(message, raw=False)
            command = data.get("command", "infer")

            if command == "reset":
                server.reset()
                await websocket.send(msgpack.packb({"status": "reset"}, use_bin_type=True))

            elif command == "infer":
                obs = data["obs"]
                return_viz = data["return_viz"]
                obs = server.process_point(obs)
                action, viz = server.infer(obs, return_viz)
                action_ee = server.point2action(action)
                await websocket.send(msgpack.packb({"action": action_ee, "viz": viz}, use_bin_type=True))

            elif command == "shutdown":
                logging.info(f"Port {args.port}: Received shutdown signal. Exiting...")
                await websocket.send(msgpack.packb({"status": "shutdown"}, use_bin_type=True))
                shutdown_event.set()
                return
            
            else:
                await websocket.send(json.dumps({"error": f"Unknown command: {command}"}))

    async def serve():
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Serving on host=%s ip=%s port=%d", hostname, local_ip, args.port)

        async with websockets.serve(handle, "0.0.0.0", args.port, max_size=100 * 1024 * 1024):  # 100MB
            await asyncio.Future()
        
        logging.info(f"Port {args.port}: Server shut down.")

    asyncio.run(serve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))