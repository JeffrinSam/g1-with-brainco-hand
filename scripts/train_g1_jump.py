"""
train_g1_jump.py

A self-contained reinforcement learning training script for the Unitree G1 humanoid robot in MuJoCo.
Features:
- Implements a custom Gymnasium environment (G1JumpEnv) wrapping the native MuJoCo scene.
- Uses a phase-based dense reward structure (Crouch -> Explosive Jump -> Flight -> Landing & Balance).
- Uses Stable-Baselines3 (PPO) to train the policy.
- Supports training checkpointing, monitoring, and visual evaluation.
"""

import os
import time
import argparse
import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class G1JumpEnv(gym.Env):
    """
    Custom Gymnasium Environment for Unitree G1 jumping and balancing.
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 50}

    def __init__(self, render_mode="none"):
        super().__init__()
        
        self.render_mode = render_mode
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.script_dir)
        self.scene_path = os.path.join(self.root_dir, "scenes", "mujoco", "g1_walk_scene.xml")
        
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        
        # Configure simulation parameters
        self.model.opt.timestep = 0.002
        self.control_decimation = 10  # 50 Hz control rate (20ms step)
        
        # Stabilize dexterous fingers under PD control
        self.model.dof_damping[6:] = 0.001
        self.model.dof_armature[6:] = 0.01
        self.model.dof_frictionloss[6:] = 0.1
        
        # Resolve joint names
        self.leg_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
        ]
        self.arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        self.hand_joint_names = [
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
        
        self.body_joint_names = self.leg_joint_names + self.arm_joint_names  # 29 joints
        
        # Resolve address mapping
        _, self.leg_qpos_adrs, self.leg_dof_adrs = self._find_joint_info(self.leg_joint_names)
        _, self.arm_qpos_adrs, self.arm_dof_adrs = self._find_joint_info(self.arm_joint_names)
        _, self.hand_qpos_adrs, self.hand_dof_adrs = self._find_joint_info(self.hand_joint_names)
        _, self.body_qpos_adrs, self.body_dof_adrs = self._find_joint_info(self.body_joint_names)
        
        # Resolve key body ids for measurements
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.left_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link") # foot representation
        self.right_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link")
        
        # Environment Settings
        self.max_episode_steps = 150  # 150 steps * 20ms = 3.0s
        self.action_scale = 0.25
        self.default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                         0.0, 0.0, 0.0], dtype=np.float32)
        
        # PD parameters for leg/waist (controlled by RL)
        self.kps = np.array([150, 150, 150, 200, 40, 40,
                             150, 150, 150, 200, 40, 40,
                             250.0, 250.0, 250.0], dtype=np.float32)
        self.kds = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                             2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                             5.0, 5.0, 5.0], dtype=np.float32)
        
        # Arm PD parameters (keep standard posture or let RL offset them slightly)
        self.arm_kps = np.full(14, 100.0, dtype=np.float32)
        self.arm_kds = np.full(14, 2.0, dtype=np.float32)
        
        # Hand PD parameters (stay neutral)
        self.hand_kps = np.full(32, 50.0, dtype=np.float32)
        self.hand_kds = np.full(32, 1.0, dtype=np.float32)
        
        # Space specs
        # Action space: target offset positions for 29 joints (15 legs/waist + 14 arms)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)
        
        # Observation space dimension:
        # - Base Z height (1)
        # - Base orientation quaternion (4)
        # - Base linear velocity (3)
        # - Base angular velocity (3)
        # - Joint positions (29)
        # - Joint velocities (29)
        # - Phase time indicator (1)
        obs_dim = 1 + 4 + 3 + 3 + 29 + 29 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.viewer = None
        self.step_counter = 0

    def _find_joint_info(self, names):
        ids, qpos_adrs, dof_adrs = [], [], []
        for name in names:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id == -1:
                raise ValueError(f"Joint '{name}' not found.")
            ids.append(j_id)
            qpos_adrs.append(self.model.jnt_qposadr[j_id])
            dof_adrs.append(self.model.jnt_dofadr[j_id])
        return ids, qpos_adrs, dof_adrs

    def _get_obs(self):
        pelvis_z = self.data.qpos[2]
        quat = self.data.qpos[3:7]  # qw, qx, qy, qz
        lin_vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]
        
        qj = np.array([self.data.qpos[adr] for adr in self.body_qpos_adrs], dtype=np.float32)
        dqj = np.array([self.data.qvel[adr] for adr in self.body_dof_adrs], dtype=np.float32)
        
        phase = self.step_counter / self.max_episode_steps
        
        return np.concatenate([
            [pelvis_z],
            quat,
            lin_vel,
            ang_vel,
            qj,
            dqj,
            [phase]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset MjData state
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial pelvis height (0.74m)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.74
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Set default joint positions
        for idx, qpos_adr in enumerate(self.leg_qpos_adrs):
            self.data.qpos[qpos_adr] = self.default_angles[idx]
            
        mujoco.mj_forward(self.model, self.data)
        
        self.step_counter = 0
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # 1. Translate action values to joint targets
        # Clip actions to stay within boundaries
        action = np.clip(action, -1.0, 1.0)
        
        # Decode target positions for leg/waist (15 joints)
        leg_targets = action[:15] * self.action_scale + self.default_angles
        # Target positions for arms (14 joints) - centered around zero
        arm_targets = action[15:] * self.action_scale
        
        # Run physics sub-stepping loop
        for _ in range(self.control_decimation):
            self.data.qfrc_applied[:] = 0.0
            
            # Leg/Waist PD Torques
            curr_leg_q = np.array([self.data.qpos[adr] for adr in self.leg_qpos_adrs], dtype=np.float32)
            curr_leg_dq = np.array([self.data.qvel[adr] for adr in self.leg_dof_adrs], dtype=np.float32)
            leg_torques = (leg_targets - curr_leg_q) * self.kps - curr_leg_dq * self.kds
            
            # Arm PD Torques
            curr_arm_q = np.array([self.data.qpos[adr] for adr in self.arm_qpos_adrs], dtype=np.float32)
            curr_arm_dq = np.array([self.data.qvel[adr] for adr in self.arm_dof_adrs], dtype=np.float32)
            arm_torques = (arm_targets - curr_arm_q) * self.arm_kps - curr_arm_dq * self.arm_kds
            
            # Hand PD Torques (Neutral zero targets)
            curr_hand_q = np.array([self.data.qpos[adr] for adr in self.hand_qpos_adrs], dtype=np.float32)
            curr_hand_dq = np.array([self.data.qvel[adr] for adr in self.hand_dof_adrs], dtype=np.float32)
            hand_torques = (0.0 - curr_hand_q) * self.hand_kps - curr_hand_dq * self.hand_kds
            
            # Apply calculated torques to DOF addresses
            for i, dof_adr in enumerate(self.leg_dof_adrs):
                self.data.qfrc_applied[dof_adr] = leg_torques[i]
            for i, dof_adr in enumerate(self.arm_dof_adrs):
                self.data.qfrc_applied[dof_adr] = arm_torques[i]
            for i, dof_adr in enumerate(self.hand_dof_adrs):
                self.data.qfrc_applied[dof_adr] = hand_torques[i]
                
            mujoco.mj_step(self.model, self.data)
            
        self.step_counter += 1
        
        # 2. State metrics
        pelvis_z = self.data.qpos[2]
        pelvis_vz = self.data.qvel[2]
        t = self.data.time
        
        # 3. Dense Phase-Based Reward Shaping
        reward = 0.0
        
        # Phase A: Crouch / Compression Phase (0.0s to 0.4s)
        # Encourages the robot to lower its base to prepare for takeoff
        if t < 0.4:
            target_crouch_height = 0.50
            reward = 3.0 * (1.0 - abs(pelvis_z - target_crouch_height) / 0.24)
            
        # Phase B: Explosive Takeoff / Thrust Phase (0.4s to 0.8s)
        # Encourages explosive upward vertical velocity
        elif t < 0.8:
            thrust_bonus = max(0.0, pelvis_vz) * 8.0
            height_bonus = max(0.0, pelvis_z - 0.74) * 15.0
            reward = thrust_bonus + height_bonus
            
        # Phase C: Flight Phase (0.8s to 1.4s)
        # Maximizes the peak height achieved during flight
        elif t < 1.4:
            flight_bonus = max(0.0, pelvis_z - 0.74) * 25.0
            reward = flight_bonus
            
        # Phase D: Landing & Stabilization / Balancing Phase (1.4s to 3.0s)
        # Encourages returning to standing height and minimizing velocities (balancing)
        else:
            h_error = abs(pelvis_z - 0.74)
            vel_error = np.linalg.norm(self.data.qvel[:6])  # penalize linear/angular velocities of base
            
            # Standing survival reward + penalty for errors
            standing_reward = max(0.0, 1.0 - h_error / 0.15) * 5.0
            stab_reward = max(0.0, 1.0 - vel_error / 3.0) * 3.0
            reward = standing_reward + stab_reward
            
        # Control regularization (penalize extreme movements and high torques)
        control_penalty = -0.01 * np.sum(np.square(action))
        reward += control_penalty
        
        # Orientation penalty (keep pelvis upright - qw close to 1)
        qw = self.data.qpos[3]
        orientation_penalty = -10.0 * (1.0 - qw**2)
        reward += orientation_penalty
        
        # 4. Termination conditions
        terminated = False
        truncated = self.step_counter >= self.max_episode_steps
        
        # Terminate if the robot falls completely (pelvis drops below 0.35m or tilts excessively)
        if pelvis_z < 0.35:
            terminated = True
            reward -= 50.0  # Large crash penalty
        elif qw < 0.70:     # Tilt exceeded ~45 degrees
            terminated = True
            reward -= 50.0
            
        obs = self._get_obs()
        
        # 5. Render if human mode is enabled
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.viewer is None:
            from mujoco.viewer import launch_passive
            self.viewer = launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -15
            self.viewer.cam.distance = 2.2
            self.viewer.cam.lookat = np.array([0.0, 0.0, 0.5])
            
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def main():
    parser = argparse.ArgumentParser(description="Train G1 Humanoid Jumping & Balancing Policy")
    parser.add_argument("--steps", type=int, default=500000,
                        help="Number of training timesteps (default: 500,000)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate the policy visually after training completes")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints",
                        help="Checkpoint directory (default: ./checkpoints)")
    args = parser.parse_args()

    print("=== G1 Humanoid Reinforcement Learning: Jumping & Balancing ===")
    print("Setting up training environment...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected for policy training: {device.upper()}")

    # Stable-Baselines3 training wrapper
    def make_env():
        return G1JumpEnv(render_mode="none")

    # Create multi-environment parallel training vector
    # We use 4 vectorized environments for fast training
    num_envs = 4
    try:
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        print(f"Created {num_envs} parallel vectorized training environments.")
    except Exception as e:
        print(f"Warning: Could not create SubprocVecEnv: {e}. Falling back to DummyVecEnv...")
        env = DummyVecEnv([make_env for _ in range(num_envs)])

    # Setup callbacks
    os.makedirs(args.checkpoint, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1000),
        save_path=args.checkpoint,
        name_prefix="g1_jump_model"
    )

    # Instantiate PPO model with optimized parameters for locomotion tasks
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="./tensorboard_logs"
    )

    print(f"Starting training run for {args.steps} steps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint_callback
        )
        print(f"\nTraining completed successfully in {time.time() - start_time:.2f} seconds!")
        
        # Save final model
        model_save_path = os.path.join(args.checkpoint, "g1_jump_final.zip")
        model.save(model_save_path)
        print(f"Saved final jumping model to: {model_save_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        model.save(os.path.join(args.checkpoint, "g1_jump_interrupted.zip"))
        
    finally:
        env.close()

    # If eval is requested, run visual check
    if args.eval:
        print("\nEvaluating trained policy visually...")
        eval_env = G1JumpEnv(render_mode="human")
        obs, _ = eval_env.reset()
        
        # Try to load final model
        try:
            trained_model = PPO.load(os.path.join(args.checkpoint, "g1_jump_final.zip"))
            print("Loaded trained model for visual evaluation.")
        except Exception:
            print("Warning: Could not load final model. Evaluating random policy.")
            trained_model = None

        print("Press Ctrl+C in the terminal to exit visual evaluation.")
        try:
            while True:
                if trained_model is not None:
                    action, _ = trained_model.predict(obs, deterministic=True)
                else:
                    action = eval_env.action_space.sample()
                    
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                
                if terminated or truncated:
                    print(f"Episode complete. Pelvis final height: {eval_env.data.qpos[2]:.4f}m. Resetting...")
                    time.sleep(1.0)
                    obs, _ = eval_env.reset()
                    
                # Sleep to run at real-time rendering speed
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nVisual evaluation closed.")
        finally:
            eval_env.close()

if __name__ == "__main__":
    main()
