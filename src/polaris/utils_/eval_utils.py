import numpy as np
from scipy.spatial.transform import Rotation
import torch
import random

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for deterministic behavior (may slow down)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def _sample_from_zones(zones: list) -> float:
    """Randomly pick one zone from a list of [min, max] pairs, then sample within it."""
    zone = zones[np.random.randint(len(zones))]   # pick a zone uniformly
    return np.random.uniform(zone[0], zone[1])     # sample within that zone


def randomize_object_poses(randomization_setting: dict, initial_conditions: list, min_distance: float = 0.05, max_retries: int = 100) -> list:
    result = dict(initial_conditions[0])
    placed_positions = []  # Track (x, y, z) of newly placed objects only

    for obj, ranges in randomization_setting.items():
        for attempt in range(max_retries):
            x = _sample_from_zones(ranges["x"])
            y = _sample_from_zones(ranges["y"])
            z = _sample_from_zones(ranges["z"])

            # Check distance against all previously placed objects
            too_close = any(
                np.linalg.norm(np.array([x, y, z]) - np.array(pos)) < min_distance
                for pos in placed_positions
            )

            if not too_close:
                break
        else:
            raise RuntimeError(
                f"Could not place '{obj}' at least {min_distance*100:.0f}cm away from "
                f"other objects after {max_retries} attempts. "
                f"Consider expanding the randomization zones."
            )

        placed_positions.append([x, y, z])

        # ori ranges are still flat [min, max] — keep as-is
        rx = np.random.uniform(*ranges.get("ori_x", [0, 0]))
        ry = np.random.uniform(*ranges.get("ori_y", [0, 0]))
        rz = np.random.uniform(*ranges.get("ori_z", [0, 0]))

        ori_pose = initial_conditions[0][obj]       # [x, y, z, qx, qy, qz, qw]
        orig_q   = Rotation.from_quat(ori_pose[3:7])
        rel_q    = Rotation.from_euler("xyz", [rx, ry, rz])
        final_q  = orig_q * rel_q
        q        = final_q.as_quat()                # xyzw

        result[obj] = [x, y, z, q[3], q[0], q[1], q[2]]  # wxyz for Isaac Sim

        print(f"[{obj}] pos=({x:.3f}, {y:.3f}, {z:.3f})  quat(wxyz)=({q[3]:.3f}, {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")

    return [result]

def is_success(info) -> bool:
    try:
        val = info["rubric"]["success"]
        v = val[0] if hasattr(val, "__len__") else val
        if isinstance(v, torch.Tensor):
            return bool(v.item())
        return bool(v)
    except (KeyError, TypeError, IndexError) as e:
        print(f"  [warn] Could not read info['rubric']['success']: {e}")
        return False