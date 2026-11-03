# import numpy as np
# import json
# import torch
# import robosuite.utils.transform_utils as T

# from polaris.policy.abstract_client import InferenceClient
# from polaris.config import PolicyArgs

# from polaris.utils_.data_utils import debug_plot
# from polaris.utils_.transform_utils import rotation_transfer_6D_to_matrix, state_quat_to_rot6d
# from eval_smith_utils import load_multitask_high_level_model, load_low_level_policy, infer_multitask_high_level_model, low_level_policy_infer

# @InferenceClient.register(client_name="Smith")
# class SmithClient(InferenceClient):

#     def __init__(self, args: PolicyArgs) -> None:
#         self.args = args
#         if args.policy_path is None:
#             raise ValueError("policy_path must be set for Smith")
#         assert type(args.policy_path) == tuple
        
#         self.hl_path, self.ll_path = args.policy_path

#         self.hl_policy, self.hl_policy_args= load_multitask_high_level_model(self.hl_path)
#         self.ll_policy, self.ll_policy_args = load_low_level_policy(self.ll_path, "latest.ckpt")

#         self.device = args.device
#         self.hl_policy.to(self.device)
#         self.ll_policy.to(self.device)

#         self.open_loop_horizon = args.open_loop_horizon
#         self.actions_from_chunk_completed = 0
#         self.pred_action_chunk = None

#         self.calibration = self.get_cam_param("/home/haotian/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json")

#         self.debug_frames = []
#         self.debug = True

#     def get_cam_param(self, calibration_path):
#         with open(calibration_path, "r") as f:
#             cams = json.load(f)

#         calibration = {}
#         for name, c in cams.items():
#             calibration[name] = {
#                 "intrinsic":  np.array(c["intrinsic"]),
#                 "extrinsic":  np.array(c["extrinsic"]),  # cam_to_base (4, 4)
#                 "distortion": np.array(c["distortion"]),
#             }

#         return calibration


#     @property
#     def rerender(self) -> bool:
#         return (
#             self.actions_from_chunk_completed == 0
#             or self.actions_from_chunk_completed >= self.open_loop_horizon
#         )

#     def reset(self):
#         self.actions_from_chunk_completed = 0
#         self.pred_action_chunk = None


#     def infer(
#         self, obs: dict, language_instruction = None, target_pose = None, return_viz=True
#     ) -> tuple[np.ndarray, np.ndarray | None]:
        
#         torch.cuda.empty_cache()
        
#         with torch.no_grad():
#             hl_obs = self._extract_hl_obs(obs) # 1, N+4, 3

#             if target_pose is None: # Debug
#                 target_pose = infer_multitask_high_level_model(hl_obs, self.hl_policy, high_level_args = self.hl_policy_args)

#             obj_pcd, agent_pos, target_pose, gripper_pcd = self._extract_ll_obs(obs, target_pose)
            
#             action = low_level_policy_infer(obj_pcd, agent_pos, target_pose, gripper_pcd, self.ll_policy) # 1, T, 10

#         env_action = self.policy_action_batch_to_env_action(obs, action) # T, 8

#         viz = None
#         if return_viz:
#             import cv2
#             ext_small  = obs["splat"]["cam1"]

#             goal_plot = debug_plot(obs["splat"]["cam1"], 
#                                     target_pose[0,0].detach().cpu().numpy(), 
#                                     self.calibration["cam1"]["intrinsic"], 
#                                     self.calibration["cam1"]["extrinsic"])
            
#             viz = np.concatenate([ext_small, goal_plot], axis=1)

#         return env_action, viz

#     def _extract_hl_obs(self, obs: dict) -> dict:

#         object_pcd = torch.from_numpy(obs["object_pcd"]).unsqueeze(0).to(self.device)  # (1, 4500, 3)
#         gripper_pcd = torch.from_numpy(obs["gripper_pcd"]).unsqueeze(0).to(self.device)

#         return torch.cat([object_pcd, gripper_pcd], dim=1) # 1, N+4, 3
    
#     def _extract_ll_obs(self, obs: dict, target_pose: torch.Tensor) -> tuple:
#         n_obs_steps = self.ll_policy_args.n_obs_steps

#         target_pose = target_pose.repeat(1, n_obs_steps, 1, 1).to(self.device)
#         agent_pos = state_quat_to_rot6d(obs["state"])
#         obj_pcd     = torch.from_numpy(obs["object_pcd"]).unsqueeze(0).unsqueeze(0).repeat(1, n_obs_steps, 1, 1).to(self.device)   # fixed typo
#         gripper_pcd = torch.from_numpy(obs["gripper_pcd"]).unsqueeze(0).unsqueeze(0).repeat(1, n_obs_steps, 1, 1).to(self.device)
#         agent_pos   = torch.from_numpy(agent_pos).unsqueeze(0).unsqueeze(0).repeat(1, n_obs_steps, 1).to(self.device)

#         return obj_pcd, agent_pos, target_pose, gripper_pcd

#     def policy_action_batch_to_env_action(self, obs, batch_action):
#         """
#         Convert a batch of policy actions to env actions.

#         Args:
#             batch_action: (1, T, 10) policy output
#             obs: current observation

#         Returns:
#             env_actions: torch.Tensor (n_action_steps, 8)
#         """
#         batch_action   = batch_action.squeeze(0)  # (T, 10)
#         n_action_steps = self.ll_policy_args.n_action_steps

#         actions = [
#             self.policy_action_to_env_action(obs, batch_action[i])  # (1, 8) np.float32
#             for i in range(n_action_steps)
#         ]

#         return torch.from_numpy(np.concatenate(actions, axis=0)).to(self.device)  # (n_action_steps, 8)


#     def policy_action_to_env_action(self, obs, action):
#         """
#         Convert a single policy action to env action.

#         Args:
#             action: (10,) tensor on self.device

#         Returns:
#             env_action: np.ndarray (1, 8) = [x, y, z, qw, qx, qy, qz, gripper]
#         """
#         # Move to CPU/numpy for all the rotation math
#         action = action.detach().cpu().numpy()

#         delta_pos     = action[:3].astype(np.float64)
#         delta_rot_6d  = action[3:9].astype(np.float64)

#         state             = obs["state"]  # numpy (8,)
        
#         cur_eef_pos       = state[:3]
#         cur_eef_quat_wxyz = state[3:7]
#         cur_eef_quat_xyzw = cur_eef_quat_wxyz[[1, 2, 3, 0]]  # wxyz → xyzw for T.quat2mat
#         cur_eef_gripper   = float(state[-1])

#         cur_R        = T.quat2mat(cur_eef_quat_xyzw)                   # (3, 3)
#         delta_R_grip = rotation_transfer_6D_to_matrix(delta_rot_6d)    # (3, 3)
#         delta_R_world = cur_R @ delta_R_grip @ cur_R.T

#         next_pos     = cur_eef_pos + delta_pos
#         next_R_world = delta_R_world @ cur_R
#         # -0.01 open; 0.01 close
#         # -1 open; 1 close
#         # 0 open; 1 close
#         next_gripper = np.clip(float(action[-1]) / 0.01, -1, 1)
#         next_gripper = (next_gripper + 1) / 2

#         next_quat_xyzw = T.mat2quat(next_R_world)           # (4,) xyzw
#         next_quat_wxyz = next_quat_xyzw[[3, 0, 1, 2]]       # → wxyz for cuRobo

#         env_action = np.concatenate([
#             next_pos,
#             next_quat_wxyz,
#             [next_gripper],
#         ], dtype=np.float32)  # (8,)

#         return env_action[None]  # (1, 8)