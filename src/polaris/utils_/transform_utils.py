import numpy as np
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

MAX_GRIPPER_WIDTH = 0.05

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (wxyz) to 6D rotation representation.
    
    Args:
        quat_wxyz (4,)  quaternion in wxyz format
    
    Returns:
        rot6d (6,)  first two columns of rotation matrix, flattened
    """
    # wxyz → xyzw for T.quat2mat
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = T.quat2mat(quat_xyzw)  # (3, 3)
    return rotation_transfer_matrix_to_6D(R).astype(np.float32)  # (6,)

def state_quat_to_rot6d(state: np.ndarray) -> np.ndarray:
    """
    Convert state from quat to rot6d representation.
    
    Args:
        state (8,)   [pos(3) | quat_wxyz(4) | gripper(1)]
    
    Returns:
        state (10,)  [pos(3) | rot6d(6) | gripper(1)]
    """
    pos     = state[:3]
    rot6d   = quat_wxyz_to_rot6d(state[3:7])
    gripper = state[7:8]
    return np.concatenate([pos, rot6d, gripper]).astype(np.float32)  # (10,)

def clip_delta_action(delta_pos: np.array, delta_local_rot: np.array, max_dpos: float = 0.05, max_drot: float = 0.5):
    delta_pos_clipped = np.clip(delta_pos, -max_dpos, max_dpos)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(delta_local_rot)
    rotvec = r.as_rotvec()                         # (3,) axis-angle
    angle  = np.linalg.norm(rotvec)
    if angle > max_drot:
        rotvec = rotvec / angle * max_drot         # scale down to max_drot
    delta_local_rot_clipped = Rotation.from_rotvec(rotvec).as_matrix()

    return delta_pos_clipped, delta_local_rot_clipped


def compute_delta_actions_smith(states_ee: np.ndarray, actions_ee: np.ndarray, clip_action: bool = False) -> np.ndarray:
    """
    THIS IS FOR SMITH data!!!!!
    
    Args:
        states_ee  (T, 8)   [pos(3) | quat_wxyz(4) | gripper(1)]
        actions_ee (T-1, 8) [pos(3) | quat_wxyz(4) | gripper(1)]  - target poses

    Returns:
        deltas (T-1, 10)  [delta_pos(3) | delta_rot6d(6) | gripper(1) normalized (-0.01, 0.01)]
    """
    T = actions_ee.shape[0]
    
    # --- delta pos (base frame) ---
    delta_pos = actions_ee[:, :3] - states_ee[:T, :3]  # (T-1, 3)

    # --- delta rot (EEF local frame) ---
    # Convert wxyz to xyzw
    cur_quat_xyzw = states_ee[:T, [4, 5, 6, 3]]      # (T-1, 4)
    action_quat_xyzw = actions_ee[:, [4, 5, 6, 3]]           # (T-1, 4)

    # Compute rotation matrices
    cur_rot = np.array([T.quat2mat(q) for q in cur_quat_xyzw])       # (T-1, 3, 3)
    action_rot = np.array([T.quat2mat(q) for q in action_quat_xyzw]) # (T-1, 3, 3)

    # Local frame delta: cur_rot.T @ action_rot
    delta_local_rot = np.einsum('nij,njk->nik', cur_rot.transpose(0, 2, 1), action_rot)  # (T-1, 3, 3)
    
    if clip_action:
        delta_pos_list = []
        delta_local_rot_list = []
        for i in range(T):
            dp, dr = clip_delta_action(delta_pos[i], delta_local_rot[i])
            delta_pos_list.append(dp)
            delta_local_rot_list.append(dr)
        delta_pos = np.stack(delta_pos_list, axis=0)
        delta_local_rot = np.stack(delta_local_rot_list, axis=0)
    
    # Convert to 6D rotation
    delta_rot_6d = np.array([rotation_transfer_matrix_to_6D(r).reshape(-1) for r in delta_local_rot])  # (T-1, 6)

    # --- normalized gripper to SMITH dataset ---
    gripper_normalized = np.where(actions_ee[:, -1] > MAX_GRIPPER_WIDTH / 2, -0.01, 0.01)  # (T-1,)

    return np.concatenate([delta_pos, delta_rot_6d, gripper_normalized[:, None]], axis=1).astype(np.float32)  # (T-1, 10)


def quat_to_rot_matrix_batch(quat_wxyz):
    """
    Convert quaternions (wxyz) to rotation matrices.
    
    Args:
        quat_wxyz: (N, 4) quaternions in [w, x, y, z] format
    
    Returns:
        (N, 3, 3) rotation matrices
    """
    # scipy uses xyzw format
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).as_matrix()


def rot_matrix_to_axis_angle_batch(rot_mat):
    """
    Convert rotation matrices to axis-angle representation.
    
    Args:
        rot_mat: (N, 3, 3) rotation matrices
    
    Returns:
        (N, 3) axis-angle vectors
    """
    return R.from_matrix(rot_mat).as_rotvec()


def compute_delta_actions_robomimic(states_ee, actions_ee, max_dpos=0.05, max_drot=0.5):
    """
    THIS IS FOR ROBOMIMIC DATA !!!
    
    Args:
        states_ee: (T, 8) = pos(3) + quat_wxyz(4) + gripper(1)
        action_ee: (T-1, 8) = pos(3) + quat_wxyz(4) + gripper(1)
        max_dpos: max position delta for normalization
        max_drot: max rotation delta for normalization
    
    Returns:
        actions: (T-1, 7) = delta_pos(3) + delta_axis_angle(3) + gripper(1)
                 where gripper is normalized to [-1, +1]
    """
    T = actions_ee.shape[0]
    
    # Current states (T-1,)
    curr_pos = states_ee[:T, :3]          # (T-1, 3)
    curr_quat = states_ee[:T, 3:7]        # (T-1, 4) wxyz
    curr_rot = quat_to_rot_matrix_batch(curr_quat)  # (T-1, 3, 3)
    
    # Target states
    target_pos = actions_ee[:, :3]         # (T-1, 3)
    target_quat = actions_ee[:, 3:7]       # (T-1, 4) wxyz
    target_rot = quat_to_rot_matrix_batch(target_quat)  # (T-1, 3, 3)
    target_gripper = actions_ee[:, -1]     # (T-1,) 0=open, 1=close
    
    # Delta position (normalized)
    delta_pos = target_pos - curr_pos     # (T-1, 3)
    delta_pos = np.clip(delta_pos / max_dpos, -1., 1.)
    
    # Delta rotation: delta_rot @ curr_rot = target_rot
    # So: delta_rot = target_rot @ curr_rot.T
    curr_rot_T = curr_rot.transpose(0, 2, 1)  # (T-1, 3, 3)
    delta_rot_mat = np.einsum('nij,njk->nik', target_rot, curr_rot_T)  # (T-1, 3, 3)
    
    # Convert to axis-angle (normalized)
    delta_axis_angle = rot_matrix_to_axis_angle_batch(delta_rot_mat)  # (T-1, 3)
    delta_axis_angle = np.clip(delta_axis_angle / max_drot, -1., 1.)
    
    # Gripper: normalize from [0,1] to [-1,+1] open close
    gripper_action = target_gripper * 2.0 - 1.0  # (T-1,)
    
    # Concatenate
    actions = np.concatenate([
        delta_pos,                    # (T-1, 3)
        delta_axis_angle,             # (T-1, 3)
        gripper_action[:, None]       # (T-1, 1)
    ], axis=1)
    
    return actions.astype(np.float32)  # (T-1, 7)
