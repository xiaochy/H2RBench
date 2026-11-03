import numpy as np
from pathlib import Path
import open3d as o3d
import time
import cv2


def make_arrow(start, end, color):
    pts = np.array([start, end])
    lines = [[0, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


folder    = Path("/home/haotian/polaris/data_smith/lift_red_cup_debug/demo_0")
output_video = str(folder / "playback.mp4")
npz_files = sorted(folder.glob("*.npz"), key=lambda p: int(p.stem))

fps = 10
dt  = 1.0 / fps
W, H = 1280, 720

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Demo Playback", width=W, height=H, visible=True)

first = np.load(npz_files[0], allow_pickle=True)
dummy = o3d.geometry.PointCloud()
dummy.points = o3d.utility.Vector3dVector(first["point_cloud"][0])
vis.add_geometry(dummy)

ctr = vis.get_view_control()
ctr.set_front([0, -1, 0])
ctr.set_up([0, 0, 1])
ctr.set_lookat([0.4, 0.0, 0.3])
ctr.set_zoom(3.0)
cam_params = ctr.convert_to_pinhole_camera_parameters()

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (W, H),
)

for npz_file in npz_files:
    data = np.load(npz_file, allow_pickle=True)

    state            = data["state"][0]
    action           = data["action"][0]
    gripper_pcd      = data["gripper_pcd"][0]
    goal_gripper_pcd = data["goal_gripper_pcd"][0]
    point_cloud      = data["point_cloud"][0]

    print(f"{npz_file.name}  pos={action[:3]}  action={action[-1]:.3f}")

    obj_o3d = o3d.geometry.PointCloud()
    obj_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    obj_o3d.paint_uniform_color([0.0, 0.0, 1.0])

    cur_o3d = o3d.geometry.PointCloud()
    cur_o3d.points = o3d.utility.Vector3dVector(gripper_pcd)
    cur_o3d.paint_uniform_color([1.0, 0.0, 0.0])

    goal_o3d = o3d.geometry.PointCloud()
    goal_o3d.points = o3d.utility.Vector3dVector(goal_gripper_pcd)
    goal_o3d.paint_uniform_color([0.0, 1.0, 0.0])

    rot6d = state[3:9]
    a1 = rot6d[:3] / np.linalg.norm(rot6d[:3])
    a2 = rot6d[3:6]
    b2 = a2 - np.dot(a2, a1) * a1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(a1, b2)
    R  = np.stack([a1, b2, b3], axis=1)

    origin   = gripper_pcd[-1]
    axis_len = 0.05
    ee_frame = [
        make_arrow(origin, origin + R[:, 0] * axis_len, [1, 0, 0]),
        make_arrow(origin, origin + R[:, 1] * axis_len, [0, 1, 0]),
        make_arrow(origin, origin + R[:, 2] * axis_len, [0, 0, 1]),
    ]

    cur_finger_line  = make_arrow(gripper_pcd[1],      gripper_pcd[2],      [1, 0, 0])
    goal_finger_line = make_arrow(goal_gripper_pcd[1], goal_gripper_pcd[2], [0, 1, 0])
    world_axis       = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    all_geoms = [
        obj_o3d, cur_o3d, goal_o3d,
        cur_finger_line, goal_finger_line,
        world_axis,
    ] + ee_frame

    vis.clear_geometries()
    for g in all_geoms:
        vis.add_geometry(g, reset_bounding_box=False)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    # Capture frame
    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)

    time.sleep(dt)

    if not vis.poll_events():
        break

writer.release()
vis.destroy_window()
print(f"Saved to {output_video}")