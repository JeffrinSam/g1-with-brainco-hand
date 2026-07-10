"""
================================================================================
  visualize_ego.py - Visualize the G1 Robot's Ego-centric Head Camera
  
  PURPOSE:
    Launches the passive viewer showing the G1 robot standing in front of the
    table with the apple.
    
    This script displays the G1 robot's point of view (ego-centric view)
    using the 'head_camera' config.
    
  RUN:
    cd D:/Personal/Mujcotutorial
    .venv/Scripts/python g1_vla_inference/visualize_ego.py
================================================================================
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. Load standalone scene assets
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SCENE_XML = os.path.join(ROOT_DIR, "scenes", "mujoco", "apple_table_scene.xml")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)

# 2. Setup standard joint addresses for home poses
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
]
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
]

def get_joint_addrs(model, names):
    qpos_adrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adrs.append(model.jnt_qposadr[jid])
    return qpos_adrs

larm_qpos = get_joint_addrs(model, LEFT_ARM_JOINT_NAMES)
leg_qpos = get_joint_addrs(model, LEG_JOINT_NAMES)

# Poses
leg_default = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], dtype=np.float64)
left_arm_home = np.array([0.71, 0.40, -0.14, -0.73, 0.68, 0.73, -0.98], dtype=np.float64)

def reset_scene():
    mujoco.mj_resetData(model, data)
    pelvis_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if pelvis_jid != -1:
        adr = model.jnt_qposadr[pelvis_jid]
        data.qpos[adr:adr+3] = [0.0, 0.0, 0.85]
        data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]
    for i, adr in enumerate(leg_qpos):
        data.qpos[adr] = leg_default[i]
    for i, adr in enumerate(larm_qpos):
        data.qpos[adr] = left_arm_home[i]
    
    # Place apple
    apple_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "apple_free")
    if apple_jid != -1:
        adr = model.jnt_qposadr[apple_jid]
        data.qpos[adr:adr+3] = [0.55, 0.0, 0.74]
        data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]
        
    mujoco.mj_forward(model, data)

def main():
    reset_scene()
    
    print("\n=== Launching Ego-centric Head Camera Viewer ===")
    print("This will open the passive viewer and lock the camera to the G1 head view.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Find the ID of the head camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        
        # Override the GUI camera setting to show the G1's head viewpoint
        if cam_id != -1:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
            print(f"Set viewer camera to fixed cam: 'head_camera' (ID={cam_id})")
        else:
            print("Warning: 'head_camera' not found, defaulting to free camera.")
            
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
