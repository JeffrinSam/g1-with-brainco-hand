"""
run_wbc_mujoco.py

Runs the G1 humanoid through a scripted stand -> walk -> waypoint-navigation
sequence in MuJoCo, using the stand.onnx / walk.onnx WBC policies via ONNX
Runtime, with PD control applied via qfrc_applied. Loads scenes/mujoco/
g1_walk_scene.xml, and PD gains / standing height / action scale / walk
speed / control decimation from config/robot_params.yaml. Logs pelvis and
fingertip trajectories to simulation_walk.log.

Usage Examples:
----------------
1. Interactive viewer (macOS needs `mjpython` in place of `python`):
       mjpython scripts/run_wbc_mujoco.py

2. Headless, faster than real time, log-only:
       python scripts/run_wbc_mujoco.py --render-mode none

3. Slow down/speed up the viewer's playback (0.5 = half speed):
       mjpython scripts/run_wbc_mujoco.py --realtime-scale 0.5
"""

import os
import time
import argparse
import collections
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

def load_robot_params(path):
    """Load robot_params.yaml (PD gains, standing height, etc.) as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def expand_group_gains(joint_names, groups):
    """Expand per-joint-group kp/kd/ki gains (from robot_params.yaml) into
    per-joint arrays, matching the order of joint_names. Group membership is
    determined by substring match against each joint's name (e.g. "hip" matches
    left_hip_pitch_joint, right_hip_roll_joint, etc.)."""
    kps, kds, kis = [], [], []
    for name in joint_names:
        for group_name, gains in groups.items():
            if group_name in name:
                kps.append(gains["kp"])
                kds.append(gains["kd"])
                kis.append(gains["ki"])
                break
        else:
            raise ValueError(f"No gain group found for joint '{name}'")
    return (np.array(kps, dtype=np.float32),
            np.array(kds, dtype=np.float32),
            np.array(kis, dtype=np.float32))

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Standard PD control law: torque = kp*(target_q - q) + kd*(target_dq - dq)."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate world-frame vector v into the frame of quaternion q (i.e. apply q's inverse)."""
    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
        + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
        + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
        + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
        + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
        + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
        + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
    ], dtype=np.float32)

def get_gravity_orientation(quat):
    """Project the world "down" vector into the body frame — tells the policy which way is down."""
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate_inverse(quat, gravity_vec)

def quat_to_yaw(quat):
    """Extract yaw angle from quaternion."""
    w, x, y, z = quat
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def get_loco_cmd(robot_pos, robot_yaw, target_pos, max_speed=0.5):
    """Compute locomotion command to steer toward target waypoint."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 0.6:  # waypoint reached
        return None, dist

    # Angle to target in world frame
    target_angle = np.arctan2(dy, dx)
    angle_error = target_angle - robot_yaw

    # Normalize to [-pi, pi]
    angle_error = (angle_error + np.pi) % (2*np.pi) - np.pi

    # Combine forward and turn in one command, scaling forward speed by alignment
    alignment = np.cos(angle_error)  # 1.0 when aligned, -1.0 when facing away
    forward = np.clip(dist * 0.6, 0.2, 0.5) * np.clip(alignment, 0.0, 1.0)
    turn = np.clip(angle_error * 0.8, -0.6, 0.6)
    return np.array([forward, 0.0, turn]), dist

# Waypoints — path through the room avoiding obstacles.
# Sofa occupies roughly x:[4.0,5.0] y:[-1.35,1.35]; coffee table occupies
# roughly x:[2.4,3.2] y:[-0.7,0.7]. Waypoints below are offset with real
# clearance from both boxes since get_loco_cmd steers in straight lines
# with no obstacle avoidance.
WAYPOINTS = [
    np.array([1.5, -1.5]),
    np.array([3.5, -1.5]),
    np.array([4.3, -1.9]),   # pass below the sofa's right arm, not into it
    np.array([3.7, -1.9]),   # clear the sofa's x-range before turning back
    np.array([1.5, -1.0]),   # return leg passes below the coffee table, not through it
    np.array([0.0,  0.0]),
]

def find_joint_info(model, names):
    """Look up each named joint's id, qpos address, and dof (velocity) address."""
    ids = []
    qpos_adrs = []
    dof_adrs = []
    for name in names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1:
            raise ValueError(f"Joint '{name}' not found in model.")
        ids.append(j_id)
        qpos_adrs.append(model.jnt_qposadr[j_id])
        dof_adrs.append(model.jnt_dofadr[j_id])
    return ids, qpos_adrs, dof_adrs

def compute_observation(data, qj, dqj, action, loco_cmd, height_cmd, rpy_cmd, config):
    """Build one 86-element WBC policy observation: 7 commands (loco_cmd,
    height, rpy) + 3 angular velocity + 3 gravity direction + 29 joint
    positions + 29 joint velocities + 15 previous action."""
    # 7 commands
    command = np.zeros(7, dtype=np.float32)
    command[:3] = loco_cmd[:3] * config["cmd_scale"]
    command[3] = height_cmd
    command[4:7] = rpy_cmd

    # base angular velocity & gravity
    quat = data.qpos[3:7].copy()
    omega = data.qvel[3:6].copy()
    omega_scaled = omega * config["ang_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)

    # joint states (29 joints)
    padded_defaults = np.zeros(29, dtype=np.float32)
    padded_defaults[:15] = config["default_angles"]
    qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
    dqj_scaled = dqj * config["dof_vel_scale"]

    single_obs = np.zeros(86, dtype=np.float32)
    single_obs[0:7] = command
    single_obs[7:10] = omega_scaled
    single_obs[10:13] = gravity_orientation
    single_obs[13:42] = qj_scaled
    single_obs[42:71] = dqj_scaled
    single_obs[71:86] = action

    return single_obs

def main():
    """Load the scene and WBC policies, then run the scripted stand/walk/
    waypoint sequence (see render-mode/walk-speed/realtime-scale flags above)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    scene_path = os.path.join(root_dir, "scenes", "mujoco", "g1_walk_scene.xml")
    stand_onnx_path = os.path.join(root_dir, "model_policy", "stand.onnx")
    walk_onnx_path = os.path.join(root_dir, "model_policy", "walk.onnx")
    robot_params_path = os.path.join(root_dir, "config", "robot_params.yaml")
    robot_params = load_robot_params(robot_params_path)

    parser = argparse.ArgumentParser(description="G1 Walk & Stand WBC Simulation in MuJoCo")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "none"],
                        help="Render mode ('human' for window visualization, 'none' for headless)")
    parser.add_argument("--walk-speed", type=float, default=robot_params["walk_speed"],
                        help=f"Walking forward speed command (m/s) (default: {robot_params['walk_speed']})")
    parser.add_argument("--realtime-scale", type=float, default=1.0,
                        help="Realtime scaling factor for visual rendering (e.g. 1.0 for real-time) (default: 1.0)")
    args = parser.parse_args()

    print("=== Principal MuJoCo Simulation: G1 Walking Controller ===")
    print(f"Loading compiled scene: {scene_path}")
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Model file not found at {scene_path}")

    # Load model and data
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Apply default joint parameters dynamically (damping, armature, frictionloss)
    # to stabilize the simulation under high-gain PD control.
    # Revolute joints start at DOF index 6 (floating base has 6 DOFs).
    model.dof_damping[6:] = 0.001
    model.dof_armature[6:] = 0.01
    model.dof_frictionloss[6:] = 0.1

    # Apply Visual Realism Upgrades
    # 1. Soft lighting on camera headlight to allow fixed lights to cast shadows naturally
    model.vis.headlight.ambient[:] = [0.15, 0.15, 0.15]
    model.vis.headlight.diffuse[:] = [0.4, 0.4, 0.4]
    model.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
    # 2. Increase shadow map texture size for sharp shadows
    model.vis.quality.shadowsize = 4096

    # 3. Dynamic robot PBR material mapping based on original URDF visual colors
    try:
        dark_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "dark")
        white_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "white")
        if dark_mat_id != -1 and white_mat_id != -1:
            print("Mapping robot geoms to high-fidelity PBR materials ('dark', 'white')...")
            for i in range(model.ngeom):
                body_id = model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                # Ensure we only map robot geoms (whose bodies have non-empty names other than "world")
                if body_name != "world" and body_name != "":
                    rgba = model.geom_rgba[i]
                    # Map to dark material if color is close to dark [0.2, 0.2, 0.2]
                    if np.allclose(rgba[:3], [0.2, 0.2, 0.2], atol=0.05):
                        model.geom_matid[i] = dark_mat_id
                    # Map to white material if color is close to white [0.7, 0.7, 0.7]
                    elif np.allclose(rgba[:3], [0.7, 0.7, 0.7], atol=0.05):
                        model.geom_matid[i] = white_mat_id
    except Exception as e:
        print(f"Warning: Could not dynamically map robot materials: {e}")

    # Configure simulation time
    model.opt.timestep = 0.002
    control_decimation = robot_params["control_decimation"]  # 10 * 2ms = 20ms (50Hz) control update rate
    print(f"Simulation timestep set to {model.opt.timestep}s")
    print(f"Control frequency set to {1.0 / (model.opt.timestep * control_decimation)}Hz")

    # Joint lists
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

    body_joint_names = leg_joint_names + arm_joint_names  # Exactly 29 body joints

    # Resolve IDs, qpos addresses, and dof addresses
    leg_joint_ids, leg_qpos_adrs, leg_dof_adrs = find_joint_info(model, leg_joint_names)
    arm_joint_ids, arm_qpos_adrs, arm_dof_adrs = find_joint_info(model, arm_joint_names)
    hand_joint_ids, hand_qpos_adrs, hand_dof_adrs = find_joint_info(model, hand_joint_names)
    body_joint_ids, body_qpos_adrs, body_dof_adrs = find_joint_info(model, body_joint_names)

    # Leg/waist PD gains, expanded from the per-joint-group values in robot_params.yaml
    leg_kps, leg_kds, leg_kis = expand_group_gains(leg_joint_names, robot_params["leg_joints"])

    # Policy Configuration
    config = {
        "cmd_scale": np.array([2.0, 2.0, 0.5], dtype=np.float32),
        "ang_vel_scale": 0.5,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.05,
        "action_scale": robot_params["action_scale"],
        "default_angles": np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                     0.0, 0.0, 0.0], dtype=np.float32),
        "kps": leg_kps,
        "kds": leg_kds,
        "kis": leg_kis,
        "height_cmd": robot_params["standing_height"],
        "rpy_cmd": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }

    # Arm and hand PD parameters (zero target targets)
    arm_kps = np.full(14, robot_params["arm_joints"]["kp"], dtype=np.float32)
    arm_kds = np.full(14, robot_params["arm_joints"]["kd"], dtype=np.float32)
    hand_kps = np.full(32, robot_params["hand_joints"]["kp"], dtype=np.float32)
    hand_kds = np.full(32, robot_params["hand_joints"]["kd"], dtype=np.float32)

    # Initialize ONNX inference sessions
    print("Initializing ONNX Inference Sessions...")
    stand_session = ort.InferenceSession(stand_onnx_path, providers=['CPUExecutionProvider'])
    walk_session = ort.InferenceSession(walk_onnx_path, providers=['CPUExecutionProvider'])

    # Set initial posture and coordinates in MjData
    data.qpos[0] = 0.0   # Pelvis X
    data.qpos[1] = 0.0   # Pelvis Y
    data.qpos[2] = config["height_cmd"]  # Pelvis Z (standing height)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity orientation quaternion

    # Set legs to default angles
    for idx, qpos_adr in enumerate(leg_qpos_adrs):
        data.qpos[qpos_adr] = config["default_angles"][idx]

    # Initialize dynamics and coordinate derivations
    mujoco.mj_forward(model, data)

    # Track base body and finger body IDs for logging
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_index_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_index_tip_Link")
    right_index_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_index_tip")

    # History tracking
    obs_history = collections.deque([np.zeros(86, dtype=np.float32)] * 6, maxlen=6)
    action = np.zeros(15, dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()

    # Pre-allocate stacked observation flat array
    obs_flat = np.zeros(516, dtype=np.float32)

    # Simulation stats & recording setup
    trajectory_log = []
    step_counter = 0
    sim_duration = 60.0  # 60.0 seconds of virtual time
    total_steps = int(sim_duration / model.opt.timestep)

    # Control commands configuration
    loco_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    waypoint_idx = 0

    print("\nRunning G1 WBC Simulation...")
    print("-" * 105)
    print(f"{'Time (s)':<10} | {'Pelvis Z (m)':<15} | {'Loco Cmd [Vx, Vy, Vyaw]':<25} | {'Left Index Z':<15} | {'Right Index Z':<15}")
    print("-" * 105)

    def sim_step():
        """One 2ms physics step: steer toward the current waypoint, run WBC
        policy inference every control_decimation steps, apply PD torques,
        advance physics, and log."""
        nonlocal step_counter, target_dof_pos, action, loco_cmd, waypoint_idx

        # 1. Steer toward the current waypoint based on pelvis position and yaw
        t = data.time
        robot_pos = data.xpos[pelvis_body_id, :2] if pelvis_body_id != -1 else data.qpos[:2]
        robot_yaw = quat_to_yaw(data.qpos[3:7])

        if waypoint_idx < len(WAYPOINTS):
            cmd, dist = get_loco_cmd(robot_pos, robot_yaw, WAYPOINTS[waypoint_idx])
            if cmd is None:
                print(f"Reached waypoint {waypoint_idx}: {WAYPOINTS[waypoint_idx]}")
                waypoint_idx += 1
            else:
                loco_cmd[:] = cmd

            if waypoint_idx < len(WAYPOINTS) and step_counter % int(2.0 / model.opt.timestep) == 0:
                print(f"Pos: {robot_pos} | WP {waypoint_idx}: {WAYPOINTS[waypoint_idx]} | dist: {dist:.2f}")
        else:
            loco_cmd[:] = [0.0, 0.0, 0.0]       # All waypoints reached, stand still

        # 2. Run WBC Policy inference at decimation rate
        if step_counter % control_decimation == 0:
            # Query body joint states
            qj = np.array([data.qpos[qpos_adr] for qpos_adr in body_qpos_adrs], dtype=np.float32)
            dqj = np.array([data.qvel[dof_adr] for dof_adr in body_dof_adrs], dtype=np.float32)

            single_obs = compute_observation(data, qj, dqj, action, loco_cmd, config["height_cmd"], config["rpy_cmd"], config)
            obs_history.append(single_obs)

            # Stack history observations
            for idx, hist_obs in enumerate(obs_history):
                obs_flat[idx * 86 : (idx + 1) * 86] = hist_obs

            # Stand policy when speed is small, else Walk policy
            is_standing = np.linalg.norm(loco_cmd) <= 0.05
            session = stand_session if is_standing else walk_session

            # Perform inference
            ort_inputs = {session.get_inputs()[0].name: obs_flat.reshape(1, 516)}
            ort_outs = session.run(None, ort_inputs)
            action = ort_outs[0].squeeze()

            # Decode target joint positions
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

        # 3. Calculate PD Control Torques
        # Clear applied forces
        data.qfrc_applied[:] = 0.0

        # Leg & Waist PD Control
        current_leg_q = np.array([data.qpos[qpos_adr] for qpos_adr in leg_qpos_adrs], dtype=np.float32)
        current_leg_dq = np.array([data.qvel[dof_adr] for dof_adr in leg_dof_adrs], dtype=np.float32)
        leg_torques = pd_control(target_dof_pos, current_leg_q, config["kps"], 0.0, current_leg_dq, config["kds"])

        # Arm PD Control (Neutral posture)
        current_arm_q = np.array([data.qpos[qpos_adr] for qpos_adr in arm_qpos_adrs], dtype=np.float32)
        current_arm_dq = np.array([data.qvel[dof_adr] for dof_adr in arm_dof_adrs], dtype=np.float32)
        arm_torques = pd_control(0.0, current_arm_q, arm_kps, 0.0, current_arm_dq, arm_kds)

        # Hand PD Control (Neutral posture)
        current_hand_q = np.array([data.qpos[qpos_adr] for qpos_adr in hand_qpos_adrs], dtype=np.float32)
        current_hand_dq = np.array([data.qvel[dof_adr] for dof_adr in hand_dof_adrs], dtype=np.float32)
        hand_torques = pd_control(0.0, current_hand_q, hand_kps, 0.0, current_hand_dq, hand_kds)

        # Apply torques to dofs
        for i, dof_adr in enumerate(leg_dof_adrs):
            data.qfrc_applied[dof_adr] = leg_torques[i]
        for i, dof_adr in enumerate(arm_dof_adrs):
            data.qfrc_applied[dof_adr] = arm_torques[i]
        for i, dof_adr in enumerate(hand_dof_adrs):
            data.qfrc_applied[dof_adr] = hand_torques[i]

        # 4. Advance physics simulation
        mujoco.mj_step(model, data)

        # 5. Robust Logging Printout (every 0.1s virtual time)
        # Only recorded/printed during the scripted sequence -- the idle-standing
        # phase afterwards can run indefinitely and shouldn't grow these unbounded.
        if t <= sim_duration:
            if step_counter % int(0.1 / model.opt.timestep) == 0:
                pel_z = data.xpos[pelvis_body_id, 2] if pelvis_body_id != -1 else 0.0
                li_z = data.xpos[left_index_tip_id, 2] if left_index_tip_id != -1 else 0.0
                ri_z = data.xpos[right_index_tip_id, 2] if right_index_tip_id != -1 else 0.0
                print(f"{data.time:<10.2f} | {pel_z:<15.5f} | {str(list(np.round(loco_cmd, 2))):<25} | {li_z:<15.5f} | {ri_z:<15.5f}")

            # 6. Record trajectory data
            pelvis_pos = data.xpos[pelvis_body_id].copy() if pelvis_body_id != -1 else np.zeros(3)
            pelvis_quat = data.qpos[3:7].copy()
            li_pos = data.xpos[left_index_tip_id].copy() if left_index_tip_id != -1 else np.zeros(3)
            ri_pos = data.xpos[right_index_tip_id].copy() if right_index_tip_id != -1 else np.zeros(3)

            trajectory_log.append({
                "time": data.time,
                "pelvis_pos": pelvis_pos,
                "pelvis_quat": pelvis_quat,
                "left_index_pos": li_pos,
                "right_index_pos": ri_pos
            })

        step_counter += 1

    # Execute simulation based on render mode
    if args.render_mode == "human":
        print("Launching MuJoCo passive viewer...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera position for a nice front view
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -15
            viewer.cam.distance = 2.2
            viewer.cam.lookat = np.array([0.0, 0.0, 0.5])

            start_time = time.time()
            sequence_done_announced = False
            # Runs until the viewer window is closed manually: the scripted
            # stand/walk/turn sequence plays out, then the robot holds an idle
            # stand (loco_cmd reverts to [0, 0, 0] in sim_step) indefinitely.
            while viewer.is_running():
                # Calculate the target simulation time based on wall-clock elapsed time
                target_sim_time = (time.time() - start_time) * args.realtime_scale

                # Step the physics until simulation time catches up to wall-clock time
                # Limit the steps per sync iteration to prevent simulation locking
                max_steps_per_sync = int(0.05 / model.opt.timestep)  # max 50ms of physics steps per sync
                steps = 0
                while data.time < target_sim_time and steps < max_steps_per_sync:
                    sim_step()
                    steps += 1

                if not sequence_done_announced and data.time >= sim_duration:
                    print("-" * 105)
                    print("Scripted sequence complete -- idle standing. Close the viewer window to exit.")
                    sequence_done_announced = True

                viewer.sync()
                # Yield CPU to keep window responsive and maintain accurate timekeeping
                time.sleep(0.001)
    else:
        print("Running in HEADLESS mode...")
        while data.time < sim_duration:
            sim_step()

    print("-" * 105)
    print("Simulation run completed successfully.")

    # Save log to CSV file
    log_path = os.path.join(root_dir, "simulation_walk.log")
    with open(log_path, "w") as f:
        f.write("time,pelvis_x,pelvis_y,pelvis_z,pelvis_qw,pelvis_qx,pelvis_qy,pelvis_qz,left_index_x,left_index_y,left_index_z,right_index_x,right_index_y,right_index_z\n")
        for entry in trajectory_log:
            f.write(f"{entry['time']:.6f},"
                    f"{entry['pelvis_pos'][0]:.6f},{entry['pelvis_pos'][1]:.6f},{entry['pelvis_pos'][2]:.6f},"
                    f"{entry['pelvis_quat'][0]:.6f},{entry['pelvis_quat'][1]:.6f},{entry['pelvis_quat'][2]:.6f},{entry['pelvis_quat'][3]:.6f},"
                    f"{entry['left_index_pos'][0]:.6f},{entry['left_index_pos'][1]:.6f},{entry['left_index_pos'][2]:.6f},"
                    f"{entry['right_index_pos'][0]:.6f},{entry['right_index_pos'][1]:.6f},{entry['right_index_pos'][2]:.6f}\n")
    print(f"Log file saved to: {log_path}")

if __name__ == "__main__":
    main()
