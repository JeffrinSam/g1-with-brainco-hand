"""
vla_env.py - VLA Environment Wrapper for G1 Humanoid in MuJoCo

Minimal Gym-like environment (reset/step/get_observation) around
scenes/mujoco/apple_table_scene.xml, providing head-camera RGB frames and
left-arm joint state as observations, and taking left-arm delta-position
actions. Used by scripts/vla_inference.py; not a standalone script — import
G1VLAEnv into your own control/inference loop.

Usage Example:
----------------
    from g1_control.vla_env import G1VLAEnv

    env = G1VLAEnv()
    obs = env.reset()                        # {"image": ..., "qpos": ...}
    obs = env.step(action_delta)              # action_delta: 7 left-arm joint deltas
"""

import os
import numpy as np
import mujoco

class G1VLAEnv:
    """VLA-facing wrapper around the apple/table MuJoCo scene: camera + left-arm state in, left-arm deltas out."""

    def __init__(self):
        """Load the scene, set up the head-camera offscreen renderer, and resolve left-arm joint addresses."""
        # Resolve scene XML path relative to the repo root (src/g1_control -> repo root)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(os.path.dirname(self.script_dir))
        self.scene_xml = os.path.join(self.root_dir, "scenes", "mujoco", "apple_table_scene.xml")
        
        if not os.path.exists(self.scene_xml):
            raise FileNotFoundError(f"Base scene XML not found at {self.scene_xml}")
            
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)
        
        # Configure off-screen camera renderer (resolution expected by VLA, typically 224x224 or 256x256)
        self.width = 640
        self.height = 480
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        
        # Resolve joint addresses
        self.left_arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        ]
        self.larm_qpos = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.left_arm_joint_names]
        
    def reset(self):
        """Reset physics state, place the robot standing at the pelvis, and return the initial observation."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set floating pelvis standing base
        pelvis_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if pelvis_jid != -1:
            adr = self.model.jnt_qposadr[pelvis_jid]
            self.data.qpos[adr:adr+3] = [0.0, 0.0, 0.85]
            self.data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]
            
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()
        
    def get_observation(self):
        """
        Returns RGB image observation from G1's camera and joint states.
        """
        # Update scene graphics buffer from the robot's own head camera view
        self.renderer.update_scene(self.data, camera="head_camera")
        rgb_image = self.renderer.render()
        
        # Read current left arm positions
        q_arm = np.array([self.data.qpos[adr] for adr in self.larm_qpos])
        
        return {
            "image": rgb_image,
            "qpos": q_arm
        }
        
    def step(self, action_delta):
        """
        Steps the simulation using delta joint positions predicted by VLA.
        """
        # Update targets
        for i, adr in enumerate(self.larm_qpos):
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + action_delta[i], -3.0, 3.0)
            
        mujoco.mj_step(self.model, self.data)
        return self.get_observation()
