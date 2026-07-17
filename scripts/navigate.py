"""
navigate.py - Waypoint navigation for G1 in living room scene

Robot follows a fixed list of waypoints (WAYPOINTS below) around the room's
furniture, using the same stand.onnx/walk.onnx WBC policies and PD control
as run_wbc_mujoco.py, but with simpler turn-then-walk steering (get_loco_cmd)
and no config file or CLI flags — always runs with the interactive viewer.

Usage Examples:
----------------
1. Run with the interactive viewer (macOS needs `mjpython` in place of `python`):
       mjpython scripts/navigate.py
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Standard PD control law: torque = kp*(target_q - q) + kd*(target_dq - dq)."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_to_yaw(quat):
    """Extract yaw angle from quaternion."""
    w, x, y, z = quat
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def get_loco_cmd(robot_pos, robot_yaw, target_pos, max_speed=0.5):
    """Compute locomotion command to steer toward target waypoint."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 0.15:  # waypoint reached
        return None, dist
    
    # Angle to target in world frame
    target_angle = np.arctan2(dy, dx)
    angle_error = target_angle - robot_yaw
    
    # Normalize to [-pi, pi]
    angle_error = (angle_error + np.pi) % (2*np.pi) - np.pi
    
    # If facing wrong direction, turn first
    if abs(angle_error) > 0.3:
        return np.array([0.0, 0.0, np.clip(angle_error * 1.5, -0.8, 0.8)]), dist
    
    # Move forward with gentle steering
    forward = np.clip(dist * 0.8, 0.1, max_speed)
    turn = np.clip(angle_error * 1.0, -0.5, 0.5)
    return np.array([forward, 0.0, turn]), dist

# Waypoints — path through the room avoiding obstacles
# Start at (0,0), go around coffee table on the right side, toward sofa area
WAYPOINTS = [
    np.array([1.5, -1.5]),   # step right to avoid coffee table
    np.array([3.5, -1.5]),   # pass coffee table on right
    np.array([4.5, -1.5]),   # near sofa, to the right
    np.array([3.5,  0.0]),   # back toward center
    np.array([1.5,  0.0]),   # return to center
    np.array([0.0,  0.0]),   # back to start
]

def main():
    """Load the scene and WBC policies, then walk the robot through WAYPOINTS
    in the interactive viewer until it's closed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    model = mujoco.MjModel.from_xml_path(os.path.join(root_dir, "scenes", "mujoco", "g1_walk_scene.xml"))
    data = mujoco.MjData(model)

    model.opt.timestep = 0.002
    control_decimation = 10

    # Load locomotion policies
    stand_session = ort.InferenceSession(os.path.join(root_dir, "model_policy", "stand.onnx"))
    walk_session = ort.InferenceSession(os.path.join(root_dir, "model_policy", "walk.onnx"))

    # Joint setup — copy from run_wbc_mujoco.py
    from run_wbc_mujoco import find_joint_info, compute_observation

    hand_joint_names = [
        "left_thumb_metacarpal_joint", "left_thumb_proximal_joint", "left_thumb_distal_joint", "left_thumb_tip_joint",
        "left_index_proximal_joint", "left_index_distal_joint", "left_index_tip_joint",
        "left_middle_proximal_joint", "left_middle_distal_joint", "left_middle_tip_joint",
        "left_ring_proximal_joint", "left_ring_distal_joint", "left_ring_tip_joint",
        "left_pinky_proximal_joint", "left_pinky_distal_joint", "left_pinky_tip_joint",
        "right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_thumb_distal_joint", "right_thumb_tip",
        "right_index_proximal_joint", "right_index_distal_joint", "right_index_tip_joint",
        "right_middle_proximal_joint", "right_middle_distal_joint", "right_middle_tip_joint",
        "right_ring_proximal_joint", "right_ring_distal_joint", "right_ring_tip_joint",
        "right_pinky_proximal_joint", "right_pinky_distal_joint", "right_pinky_tip_joint"
    ]
    hand_joint_ids, hand_qpos_adrs, hand_dof_adrs = find_joint_info(model, hand_joint_names)
    hand_kps = np.full(32, 50.0, dtype=np.float32)
    hand_kds = np.full(32, 1.0, dtype=np.float32)

    leg_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
    ]
    arm_joint_names = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]

    leg_joint_ids, leg_qpos_adrs, leg_dof_adrs = find_joint_info(model, leg_joint_names)
    arm_joint_ids, arm_qpos_adrs, arm_dof_adrs = find_joint_info(model, arm_joint_names)
    body_joint_names = leg_joint_names + arm_joint_names
    body_joint_ids, body_qpos_adrs, body_dof_adrs = find_joint_info(model, body_joint_names)

    config = {
        "cmd_scale": np.array([2.0, 2.0, 0.5], dtype=np.float32),
        "ang_vel_scale": 0.5,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.05,
        "action_scale": 0.25,
        "default_angles": np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                     0.0, 0.0, 0.0], dtype=np.float32),
        "kps": np.array([150,150,150,200,40,40,150,150,150,200,40,40,250,250,250], dtype=np.float32),
        "kds": np.array([2,2,2,4,2,2,2,2,2,4,2,2,5,5,5], dtype=np.float32),
        "height_cmd": 0.74,
        "rpy_cmd": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }
    arm_kps = np.full(14, 100.0, dtype=np.float32)
    arm_kds = np.full(14, 2.0, dtype=np.float32)

    # Initialize
    import collections
    data.qpos[2] = 0.74
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for idx, adr in enumerate(leg_qpos_adrs):
        data.qpos[adr] = config["default_angles"][idx]
    mujoco.mj_forward(model, data)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    obs_history = collections.deque([np.zeros(86, dtype=np.float32)] * 6, maxlen=6)
    obs_flat = np.zeros(516, dtype=np.float32)
    action = np.zeros(15, dtype=np.float32)
    loco_cmd = np.zeros(3, dtype=np.float32)
    step_counter = 0
    waypoint_idx = 0

    print("=== G1 Navigation ===")
    print(f"Waypoints: {len(WAYPOINTS)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 8.0
        viewer.cam.lookat = np.array([2.0, 0.0, 0.5])

        start_time = time.time()
        while viewer.is_running():
            target_time = (time.time() - start_time)
            while data.time < target_time:

                # Get robot state
                robot_pos = data.xpos[pelvis_id, :2]
                robot_yaw = quat_to_yaw(data.qpos[3:7])

                # Navigate to current waypoint
                if waypoint_idx < len(WAYPOINTS):
                    cmd, dist = get_loco_cmd(robot_pos, robot_yaw, WAYPOINTS[waypoint_idx])
                    if cmd is None:
                        print(f"Reached waypoint {waypoint_idx}: {WAYPOINTS[waypoint_idx]}")
                        waypoint_idx += 1
                    else:
                        loco_cmd[:] = cmd
                else:
                    loco_cmd[:] = [0.0, 0.0, 0.0]
                    print("All waypoints reached. Standing.")

                # WBC policy inference
                if step_counter % control_decimation == 0:
                    qj = np.array([data.qpos[a] for a in body_qpos_adrs], dtype=np.float32)
                    dqj = np.array([data.qvel[a] for a in body_dof_adrs], dtype=np.float32)
                    single_obs = compute_observation(data, qj, dqj, action, loco_cmd,
                                                     config["height_cmd"], config["rpy_cmd"], config)
                    obs_history.append(single_obs)
                    for idx, h in enumerate(obs_history):
                        obs_flat[idx*86:(idx+1)*86] = h
                    is_standing = np.linalg.norm(loco_cmd) <= 0.05
                    session = stand_session if is_standing else walk_session
                    ort_inputs = {session.get_inputs()[0].name: obs_flat.reshape(1, 516)}
                    action[:] = session.run(None, ort_inputs)[0].squeeze()

                # PD control
                data.qfrc_applied[:] = 0.0
                curr_leg_q = np.array([data.qpos[a] for a in leg_qpos_adrs], dtype=np.float32)
                curr_leg_dq = np.array([data.qvel[a] for a in leg_dof_adrs], dtype=np.float32)
                target_dof = action * config["action_scale"] + config["default_angles"]
                leg_torques = pd_control(target_dof, curr_leg_q, config["kps"], 0.0, curr_leg_dq, config["kds"])
                curr_arm_q = np.array([data.qpos[a] for a in arm_qpos_adrs], dtype=np.float32)
                curr_arm_dq = np.array([data.qvel[a] for a in arm_dof_adrs], dtype=np.float32)
                arm_torques = pd_control(0.0, curr_arm_q, arm_kps, 0.0, curr_arm_dq, arm_kds)
                curr_hand_q = np.array([data.qpos[a] for a in hand_qpos_adrs], dtype=np.float32)
                curr_hand_dq = np.array([data.qvel[a] for a in hand_dof_adrs], dtype=np.float32)
                hand_torques = pd_control(0.0, curr_hand_q, hand_kps, 0.0, curr_hand_dq, hand_kds)
                for i, a in enumerate(leg_dof_adrs):
                    data.qfrc_applied[a] = leg_torques[i]
                for i, a in enumerate(arm_dof_adrs):
                    data.qfrc_applied[a] = arm_torques[i]
                for i, a in enumerate(hand_dof_adrs):
                    data.qfrc_applied[a] = hand_torques[i]

                mujoco.mj_step(model, data)
                step_counter += 1

            viewer.sync()
            time.sleep(0.001)

if __name__ == "__main__":
    main()