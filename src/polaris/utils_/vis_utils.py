import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation

def to_pt(arr):
    """Convert numpy array to OpenCV-compatible integer tuple."""
    return (int(arr[0]), int(arr[1]))

def debug_plot(frame, pcd,  K, T_cam_to_base, ee_pose=None):
    img = frame.copy()
    T_base_to_cam = np.linalg.inv(T_cam_to_base)
    pcd = pcd.squeeze(0).cpu().numpy()  # [4, 3]

    def project(pts_world):
        pts_h   = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)  # (N, 4)
        
        pts_cam = (T_base_to_cam @ pts_h.T).T[:, :3]   # (N, 3)

        pts2d = (K @ pts_cam.T).T                           # (N, 3)
        pts2d = (pts2d[:, :2] / pts2d[:, 2:3]).astype(int) # (N, 2)
        return pts2d

    # ── gripper PCD ───────────────────────────────────────────────────────────
    pts2d  = project(pcd)
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    labels = ["0", "1", "2", "3"]
    for pt, color, label in zip(pts2d, colors, labels):
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
            cv2.putText(img, label, (int(pt[0]) + 8, int(pt[1]) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if all(0 <= pts2d[i][0] < img.shape[1] and 0 <= pts2d[i][1] < img.shape[0] for i in [1, 2]):
        cv2.line(img, tuple(pts2d[1]), tuple(pts2d[2]), (255,255,255), 1)

    # ── EE pose axes ──────────────────────────────────────────────────────────
    if ee_pose is not None:
        if torch.is_tensor(ee_pose):
            ee_pose = ee_pose.cpu().numpy()
        ee_pose = ee_pose.flatten()

        ee_pos  = ee_pose[:3]                                        # xyz
        ee_quat = ee_pose[3:7]                                       # wxyz
        quat_xyzw = ee_quat[[1, 2, 3, 0]]                           # → xyzw for scipy
        R_ee = Rotation.from_quat(quat_xyzw).as_matrix()            # (3,3)

        axis_len = 0.05
        ee_axes = np.array([
            ee_pos,
            ee_pos + R_ee @ np.array([axis_len, 0, 0]),  # X → red
            ee_pos + R_ee @ np.array([0, axis_len, 0]),  # Y → green
            ee_pos + R_ee @ np.array([0, 0, axis_len]),  # Z → blue
        ])
        apts = project(ee_axes)
        o = tuple(apts[0])
        h, w = img.shape[:2]
        ee_ax_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        ee_ax_labels = ["eX", "eY", "eZ"]
        for j, (ac, al) in enumerate(zip(ee_ax_colors, ee_ax_labels)):
            ep = tuple(apts[j + 1])
            if all(0 <= p[0] < w and 0 <= p[1] < h for p in [o, ep]):
                cv2.arrowedLine(img, o, ep, ac, 2, tipLength=0.2)
                cv2.putText(img, al, ep, cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)

        # EE position dot
        if 0 <= o[0] < w and 0 <= o[1] < h:
            cv2.circle(img, o, 6, (255, 255, 255), -1)
            cv2.putText(img, "EE", (o[0] + 8, o[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── world origin axes ─────────────────────────────────────────────────────
    axis_len = 0.1
    origins = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ])
    apts = project(origins)
    o = (int(apts[0][0]), int(apts[0][1]))
    ax_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    ax_labels = ["X", "Y", "Z"]
    for j, (ac, al) in enumerate(zip(ax_colors, ax_labels)):
        ep = (int(apts[j+1][0]), int(apts[j+1][1]))
        h, w = img.shape[:2]
        if all(0 <= p[0] < w and 0 <= p[1] < h for p in [o, ep]):
            cv2.arrowedLine(img, o, ep, ac, 2, tipLength=0.2)
            cv2.putText(img, al, ep, cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)

    return img