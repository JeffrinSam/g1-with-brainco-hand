import numpy as np
import mujoco

def get_lidar_readings(model, data, pelvis_id, n_rays=16, max_range=5.0):
    """
    Shoot N rays horizontally from robot pelvis in a circle.
    Returns array of N distances normalized to [0, 1].
    """
    pelvis_pos = data.xpos[pelvis_id].copy()
    pelvis_pos[2] += 0.5  # shoot from torso height

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    distances = np.full(n_rays, max_range)

    geomid = np.array([-1], dtype=np.int32)

    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        dist = mujoco.mj_ray(
            model, data,
            pelvis_pos, direction,
            None, 1, pelvis_id,
            geomid
        )
        if dist >= 0:
            distances[i] = min(dist, max_range)

    return distances / max_range