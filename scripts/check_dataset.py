import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


# Copy from check_dataset.py from SMITH

# def rotation_transfer_6D_to_matrix(orient):
#     if type(orient) == list or type(orient) == tuple:
#         orient = np.array(orient, dtype=np.float64)

#     orient = orient.reshape(2, 3)
#     a1 = orient[0]
#     a2 = orient[1]

#     b1 = a1 / np.linalg.norm(a1)
#     b2 = a2 - np.dot(a2, b1) * b1
#     b2 = b2 / np.linalg.norm(b2)
#     b3 = np.cross(b1, b2)

#     rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

#     return rotate_matrix

# data_path = "/home/haotian/polaris/data_smith/lift_red_cup_50/demo_0/0.npz"
# data = np.load(data_path, allow_pickle=True)
# pointcloud = data['point_cloud'][:][0].astype(np.float32)
# gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
# goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)

# action = data['action'][:][0].astype(np.float32)
# state = data['state'][:][0].astype(np.float32)

# if state.shape == (8,):
#     quat = state[3:7]
#     r = Rotation.from_quat(list(quat))
#     matrix = r.as_matrix()    
# else:
#     rot6d = state[3:9]
#     matrix = rotation_transfer_6D_to_matrix(rot6d)

# print(state)
# print(action)
# data_path_2 = "/home/haotian/polaris/data_smith/lift_red_cup_50/demo_0/1.npz"
# data_2 = np.load(data_path_2, allow_pickle=True)
# pointcloud = data_2['point_cloud'][:][0].astype(np.float32)
# gripper_pcd = data_2['gripper_pcd'][:][0].astype(np.float32)
# goal_gripper_pcd = data_2['goal_gripper_pcd'][:][0].astype(np.float32)

# action_2 = data_2['action'][:][0].astype(np.float32)
# state_2 = data_2['state'][:][0].astype(np.float32)

# print(state_2)
# # print(gripper_pcd)
# # print(goal_gripper_pcd)
# breakpoint()
# obj_pcd_np = data['point_cloud'].reshape(-1, 3)
# print(obj_pcd_np.mean(axis=0))
# gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
# goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)

# # After computing `matrix` and loading gripper_pcd_np ...

# origin = gripper_pcd_np[-1]  # last point as origin

# axis_len = 0.05  # tweak as needed
# x_axis = origin + matrix[:, 0] * axis_len  # R col 0 = local X
# y_axis = origin + matrix[:, 1] * axis_len  # R col 1 = local Y
# z_axis = origin + matrix[:, 2] * axis_len  # R col 2 = local Z

# def make_arrow(start, end, color):
#     pts = np.array([start, end])
#     lines = [[0, 1]]
#     ls = o3d.geometry.LineSet()
#     ls.points = o3d.utility.Vector3dVector(pts)
#     ls.lines = o3d.utility.Vector2iVector(lines)
#     ls.colors = o3d.utility.Vector3dVector([color])
#     return ls

# gripper_frame = [
#     make_arrow(origin, x_axis, [1, 0, 0]),  # X = red
#     make_arrow(origin, y_axis, [0, 1, 0]),  # Y = green
#     make_arrow(origin, z_axis, [0, 0, 1]),  # Z = blue
# ]

# world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


# ### plot them together with different colors
# obj_pcd = o3d.geometry.PointCloud()
# obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_np)

# gripper_pcd = o3d.geometry.PointCloud()
# gripper_pcd.points = o3d.utility.Vector3dVector(gripper_pcd_np)

# goal_gripper_pcd = o3d.geometry.PointCloud()
# goal_gripper_pcd.points = o3d.utility.Vector3dVector(goal_gripper_pcd_np)

# ### set to different colors
# obj_pcd.paint_uniform_color([0.0, 0.0, 1.0])
# gripper_pcd.paint_uniform_color([1.0, 0.0, 0.0])
# goal_gripper_pcd.paint_uniform_color([0.0, 1.0, 0.0])

# # Add world origin axis
# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# print("=== Training Data ===")
# print(f"state: {state}")
# if state.shape == (8,):
#     quat_wxyz = state[3:7]  # check if this is wxyz or xyzw
#     r = Rotation.from_quat(quat_wxyz[[1,2,3,0]])  # assume wxyz → xyzw
# else:
#     rot6d = state[3:9]
#     r = Rotation.from_matrix(rotation_transfer_6D_to_matrix(rot6d))

# print(f"EE euler (xyz deg): {r.as_euler('xyz', degrees=True)}")
# print(f"action delta_pos:   {action[:3]}")
# print(f"action delta_rot6d: {action[3:9]}")
# print(f"action gripper:     {action[9]}")

# # Check what the delta rotation looks like
# delta_R = rotation_transfer_6D_to_matrix(action[3:9].astype(np.float64))
# delta_euler = Rotation.from_matrix(delta_R).as_euler('xyz', degrees=True)
# print(f"delta_R euler (xyz deg): {delta_euler}")

# # Next state
# print("\n=== Next State ===")
# if state_2.shape == (8,):
#     r2 = Rotation.from_quat(state_2[3:7][[1,2,3,0]])
# else:
#     r2 = Rotation.from_matrix(rotation_transfer_6D_to_matrix(state_2[3:9]))
# print(f"EE euler (xyz deg): {r2.as_euler('xyz', degrees=True)}")

# # Verify: apply action to state → should match state_2
# cur_R   = r.as_matrix()
# delta_R_grip  = rotation_transfer_6D_to_matrix(action[3:9].astype(np.float64))
# next_R  = cur_R @ delta_R_grip
# next_euler = Rotation.from_matrix(next_R).as_euler('xyz', degrees=True)
# print(f"\n=== Verification ===")
# print(f"predicted next euler: {next_euler}")
# print(f"actual    next euler: {r2.as_euler('xyz', degrees=True)}")
# print(f"diff (deg):           {next_euler - r2.as_euler('xyz', degrees=True)}")



import numpy as np
from pathlib import Path
import argparse


folder = Path("/home/haotian/polaris/data_smith/lift_red_cup_50/demo_0") # Path("/home/haotian/RoboGen-sim2real/SMITH_data/square_d2_correct/demo_1") # Path("/home/haotian/RoboGen-sim2real/SMITH_data/square_d2_correct/demo_1") # Path("/home/haotian/RoboGen-sim2real/SMITH_data/square_d2_correct/demo_1") Path("/home/haotian/polaris/data_smith/lift_red_cup_50/demo_0")
npz_files = sorted(folder.glob("*.npz"), key=lambda p: int(p.stem))

for npz_file in npz_files:
    data  = np.load(npz_file, allow_pickle=True)
    state = data["state"]
    action = data["action"]

    print(f"{npz_file.name}_state last={state[0,-1]}")
    print(f"{npz_file.name}_action last={action[0,-1]}")


# o3d.visualization.draw_geometries(
#     [obj_pcd, gripper_pcd, goal_gripper_pcd, world_axis] + gripper_frame
# )