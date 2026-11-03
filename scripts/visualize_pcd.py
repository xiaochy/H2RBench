import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/haotian/polaris/frame_0010_cam0.ply")
print(f"Points: {len(pcd.points)}")
print(f"Has colors: {pcd.has_colors()}")

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
o3d.visualization.draw_geometries([pcd, origin])