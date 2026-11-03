import numpy as np
import os
import json
import imageio.v3 as iio
import torch
import av

from polaris.utils_.vis_utils import debug_plot
from polaris.utils_.transform_utils import compute_delta_actions_robomimic

from scipy.ndimage import distance_transform_edt

MAX_DEPTH = 3  # in meters, for uint16 encoding

def fill_depth(depth: np.ndarray) -> np.ndarray:
    # depth: (H, W) float32
    invalid = depth == 0
    if not invalid.any():
        return depth
    depth = depth.copy()
    depth[invalid] = MAX_DEPTH
    return depth

def save_depth_mkv(depth_frames: np.ndarray, path: str, fps: int):
    """
    depth_frames: (T, H, W, 1) float32 in meters
    """
    depth = depth_frames.squeeze(-1).copy()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.stack([fill_depth(d) for d in depth])  # fill per frame
    depth = np.clip(depth, 0.0, 65.535)
    depth_mm = (depth * 1000).astype(np.uint16)
    T, H, W = depth_mm.shape

    with av.open(path, "w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "gray16le"

        for frame_data in depth_mm:
            frame = av.VideoFrame.from_ndarray(frame_data, format="gray16le")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

class ObsRecorder:
    def __init__(self, calibration_path: str, save_dir: str, fps: int = 30, ep_idx: int = 0, debug : bool = True):

        self.save_dir = save_dir
        self.ep_idx = ep_idx
        self.fps = fps

        self.next_event_idx = []
        self.all_obs = []
        self.calibration = self.get_cam_param(calibration_path)

        self.debug_frames = []
        self.debug = debug

    def get_cam_param(self, calibration_path):
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

    def add(self, obs):
  
        self.all_obs.append(obs)

    def save_subgoal(self):
        self.next_event_idx.append(len(self.all_obs) - 1)
        print(f"  Subgoal reached at frame {len(self.all_obs) - 1}")

    def process_obs(self):
        if not self.next_event_idx:
            print("[warn] no subgoals recorded, skipping goal_gripper_pcd")
            return
        

        for idx in range(len(self.all_obs)): 
            
            if self.debug:
                self.debug_frames.append(debug_plot(self.all_obs[idx]["splat"]["cam1"], 
                                                    self.all_obs[idx]["policy"]["gripper_pcd"], 
                                                    self.calibration["cam1"]["intrinsic"], 
                                                    self.calibration["cam1"]["extrinsic"],
                                                    self.all_obs[idx]["policy"]["ee_pose"]))

            # find the next event index >= idx
            next_event_idx = next(
                (self.next_event_idx[i] for i in range(len(self.next_event_idx))
                if self.next_event_idx[i] > idx),
                self.next_event_idx[-1]
            )
            self.all_obs[idx]["policy"]["goal_gripper_pcd"] = self.all_obs[next_event_idx]["policy"]["gripper_pcd"]


    def save_episode(self):
        self.process_obs()
        ep_dir = os.path.join(self.save_dir, f"episode_{self.ep_idx:06d}")
        os.makedirs(ep_dir, exist_ok=True)

        # Collect arrays from all_obs (skip first obs which has no action)
        all_obs = self.all_obs
        # N = T+1 states (includes terminal state with no action)
        N = len(all_obs)

        for cam in self.calibration.keys():
            frames  = np.stack([o["splat"][cam] for o in all_obs if o.get("splat") is not None])
            depths = np.stack([o["splat"][f"{cam}_depth"] for o in all_obs if o.get("splat") is not None])
            iio.imwrite(os.path.join(ep_dir, f"{cam}.mp4"), frames, fps=self.fps)
            save_depth_mkv(depths, os.path.join(ep_dir, f"{cam}_depth.mkv"), self.fps)
        

        if self.debug and self.debug_frames:
            debug_frames = np.stack(self.debug_frames) 
            iio.imwrite(os.path.join(ep_dir, "debug_video.mp4"), debug_frames, fps=self.fps)

        else:
            wrist_frames  = np.stack([o["splat"]["wrist_cam"] for o in all_obs]) # T
            iio.imwrite(os.path.join(ep_dir, "wrist_cam.mp4"), wrist_frames, fps=self.fps)
            
            wrist_depth = np.stack([o["splat"]["wrist_cam_depth"] for o in all_obs])
            save_depth_mkv(wrist_depth, os.path.join(ep_dir, "wrist_cam_depth.mkv"), self.fps)

            states_ee        = torch.cat([o["policy"]["ee_pose"] for o in all_obs]).cpu().numpy() # (T, 8)
            states_joint     = torch.cat([torch.cat([o["policy"]["arm_joint_pos"], o["policy"]["gripper_pos"]], dim=1) for o in all_obs]).cpu().numpy() # (T, 8)
            
            action_ee   = torch.cat([o["policy"]["action_ee"]  for o in all_obs if "action_ee"in o["policy"]]).cpu().numpy() # (T-1, 8)
            action_joint= torch.cat([o["policy"]["action_joint"] for o in all_obs if "action_joint" in o["policy"]]).cpu().numpy() # (T-1, 8)

            gripper_pcd  = torch.cat([o["policy"]["gripper_pcd"] for o in all_obs]).cpu().numpy() # (T, 4, 3)
            goal_gripper_pcd     = torch.cat([o["policy"]["goal_gripper_pcd"] for o in all_obs]).cpu().numpy() # (T, 4, 3)

            gripper_width = torch.cat([o["policy"]["gripper_width"] for o in all_obs]).cpu().numpy() # (T, 1)

            delta_action = compute_delta_actions_robomimic(states_ee, action_ee) # (T-1, 7)
            
            np.savez(
                os.path.join(ep_dir, "trajectory.npz"),
                states_ee        = states_ee.astype(np.float32),
                states_joint   = states_joint.astype(np.float32),
                action_ee      = action_ee.astype(np.float32),
                action_joint   = action_joint.astype(np.float32),
                gripper_pcd  = gripper_pcd.astype(np.float32),
                goal_gripper_pcd     = goal_gripper_pcd.astype(np.float32),
                gripper_width  = gripper_width.astype(np.float32),
                delta_action = delta_action.astype(np.float32),

            )

            print(f"  [saved] {ep_dir}/")
            print(f"           states_ee       : {states_ee.shape}")
            print(f"           states_joint    : {states_joint.shape}")
            print(f"           action_ee       : {action_ee.shape}")
            print(f"           action_joint    : {action_joint.shape}")
            print(f"           gripper_pcds : {gripper_pcd.shape}")
            print(f"           goal_gripper_pcds    : {goal_gripper_pcd.shape}")


    def reset(self):
        self.next_event_idx = []
        self.all_obs = []

        self.debug_frames = []