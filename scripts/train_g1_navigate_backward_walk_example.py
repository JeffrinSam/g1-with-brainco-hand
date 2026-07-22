"""
train_g1_navigate_backward_walk_example.py

ILLUSTRATION FILE — preserved on purpose, not meant to be trained "for real."

This is a snapshot of train_g1_navigate.py's reward function *before* the
heading-alignment term was added, kept here as a worked example of reward
hacking / specification gaming: _compute_reward() below only rewards
Euclidean distance-to-goal decreasing, with no term for which direction the
robot is facing. Trained under this reward, the PPO policy learned to walk
BACKWARD toward goals instead of turning to face them first -- a perfectly
valid way to minimize this exact reward function, just not the intended
"walk like a person" behavior, and one that caused the robot to shuffle
into walls (episodes ending pinned against wall_north at x=6.0) rather than
navigate around them.

Compare this file's _compute_reward() against the current one in
train_g1_navigate.py, which adds:

    pos = self.data.xpos[self.pelvis_id, :2]
    goal_vec = self.goal - pos
    goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
    heading_error = goal_angle - self._get_yaw()
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    heading_alignment = np.cos(heading_error)
    reward += heading_alignment * 0.3

That one addition (+1.0 when facing the goal, -1.0 when facing away) made
forward-walking-after-turning the reward-maximizing strategy instead of
backward-shuffling, and fixed the wall-collision behavior in practice.

Everything else in this file (WBC control loop, joint setup, observation/
action spaces) is identical to train_g1_navigate.py -- only _compute_reward()
differs. Not committed/pushed; local illustration only.
"""

import os
import sys
import collections
import numpy as np
import mujoco
import gymnasium
import onnxruntime as ort
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from g1_control.lidar import get_lidar_readings
from run_wbc_mujoco import pd_control, compute_observation, find_joint_info

ROOT = os.path.join(os.path.dirname(__file__), '..')
SCENE_XML = os.path.join(ROOT, 'scenes', 'mujoco', 'g1_walk_scene.xml')
STAND_ONNX_PATH = os.path.join(ROOT, 'model_policy', 'stand.onnx')
WALK_ONNX_PATH = os.path.join(ROOT, 'model_policy', 'walk.onnx')

N_LIDAR_RAYS = 16
MAX_LIDAR_RANGE = 5.0
GOAL_POSITIONS = [
    np.array([3.0, -1.5]),
    np.array([2.0,  1.0]),
    np.array([0.5, -2.0]),
]

class G1NavigationEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.control_decimation = 10  # physics steps per RL step
        self.max_episode_steps = 500
        self._step_count = 0

        # Stabilize the simulation under high-gain PD control (same as run_wbc_mujoco.py).
        # Revolute joints start at DOF index 6 (floating base has 6 DOFs).
        self.model.dof_damping[6:] = 0.001
        self.model.dof_armature[6:] = 0.01
        self.model.dof_frictionloss[6:] = 0.1

        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # Joint lists (identical to run_wbc_mujoco.py)
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
        self.body_joint_names = self.leg_joint_names + self.arm_joint_names  # Exactly 29 body joints

        _, self.leg_qpos_adrs, self.leg_dof_adrs = find_joint_info(self.model, self.leg_joint_names)
        _, self.arm_qpos_adrs, self.arm_dof_adrs = find_joint_info(self.model, self.arm_joint_names)
        _, self.hand_qpos_adrs, self.hand_dof_adrs = find_joint_info(self.model, self.hand_joint_names)
        _, self.body_qpos_adrs, self.body_dof_adrs = find_joint_info(self.model, self.body_joint_names)

        # WBC policy configuration (identical to run_wbc_mujoco.py)
        self.config = {
            "cmd_scale": np.array([2.0, 2.0, 0.5], dtype=np.float32),
            "ang_vel_scale": 0.5,
            "dof_pos_scale": 1.0,
            "dof_vel_scale": 0.05,
            "action_scale": 0.25,
            "default_angles": np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                         0.0, 0.0, 0.0], dtype=np.float32),
            "kps": np.array([150, 150, 150, 200, 40, 40,
                             150, 150, 150, 200, 40, 40,
                             250.0, 250.0, 250.0], dtype=np.float32),
            "kds": np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                             2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                             5.0, 5.0, 5.0], dtype=np.float32),
            "height_cmd": 0.74,
            "rpy_cmd": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

        # Arm and hand PD parameters (zero target posture, same as run_wbc_mujoco.py)
        self.arm_kps = np.full(14, 100.0, dtype=np.float32)
        self.arm_kds = np.full(14, 2.0, dtype=np.float32)
        self.hand_kps = np.full(32, 50.0, dtype=np.float32)
        self.hand_kds = np.full(32, 1.0, dtype=np.float32)

        # WBC ONNX inference sessions
        self.stand_session = ort.InferenceSession(STAND_ONNX_PATH, providers=['CPUExecutionProvider'])
        self.walk_session = ort.InferenceSession(WALK_ONNX_PATH, providers=['CPUExecutionProvider'])

        # WBC runtime state, (re)initialized per-episode in reset()
        self.step_counter = 0
        self.loco_cmd = np.zeros(3, dtype=np.float32)
        self.wbc_action = np.zeros(15, dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.obs_history = collections.deque([np.zeros(86, dtype=np.float32)] * 6, maxlen=6)
        self.obs_flat = np.zeros(516, dtype=np.float32)

        # Observation: [robot_x, robot_y, yaw, vx, vy,
        #               goal_dx, goal_dy, goal_dist,
        #               lidar x N_LIDAR_RAYS]
        obs_dim = 8 + N_LIDAR_RAYS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # Action: [vx, vy, yaw_rate] — locomotion command
        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.5, -1.0]),
            high=np.array([ 1.0,  0.5,  1.0]),
            dtype=np.float32
        )

        self.goal = GOAL_POSITIONS[0].copy()
        self._prev_dist_to_goal = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set standing pose
        self.data.qpos[2] = 0.74
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Set legs to default angles (same as run_wbc_mujoco.py) so the WBC
        # policy starts from the posture it was trained around.
        for idx, qpos_adr in enumerate(self.leg_qpos_adrs):
            self.data.qpos[qpos_adr] = self.config["default_angles"][idx]

        mujoco.mj_forward(self.model, self.data)

        # Randomize goal
        self.goal = GOAL_POSITIONS[
            np.random.randint(len(GOAL_POSITIONS))
        ].copy()

        # Reset WBC runtime state for the new episode
        self.step_counter = 0
        self.loco_cmd[:] = 0.0
        self.wbc_action[:] = 0.0
        self.target_dof_pos = self.config["default_angles"].copy()
        self.obs_history = collections.deque([np.zeros(86, dtype=np.float32)] * 6, maxlen=6)

        self._step_count = 0
        obs = self._get_obs()
        self._prev_dist_to_goal = self._dist_to_goal()
        return obs, {}

    def step(self, action):
        # The RL action *is* the locomotion command [vx, vy, yaw_rate] that
        # drives the WBC policy — same role loco_cmd plays in
        # run_wbc_mujoco.py, just sourced from the RL agent instead of
        # waypoint steering.
        self.loco_cmd[:] = np.clip(action, self.action_space.low, self.action_space.high)

        for _ in range(self.control_decimation):
            self._wbc_substep()

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = self._step_count >= self.max_episode_steps

        self._prev_dist_to_goal = self._dist_to_goal()
        return obs, reward, terminated, truncated, {}

    def _wbc_substep(self):
        """One physics step of the WBC control loop, transplanted from
        run_wbc_mujoco.py's sim_step() (steps 2-4: policy inference at
        decimation rate, PD control, physics step). Step 1 (waypoint
        steering) is replaced — self.loco_cmd is set directly by the RL
        action in step() instead."""

        # Run WBC policy inference at decimation rate
        if self.step_counter % self.control_decimation == 0:
            qj = np.array([self.data.qpos[adr] for adr in self.body_qpos_adrs], dtype=np.float32)
            dqj = np.array([self.data.qvel[adr] for adr in self.body_dof_adrs], dtype=np.float32)

            single_obs = compute_observation(
                self.data, qj, dqj, self.wbc_action, self.loco_cmd,
                self.config["height_cmd"], self.config["rpy_cmd"], self.config
            )
            self.obs_history.append(single_obs)

            for idx, hist_obs in enumerate(self.obs_history):
                self.obs_flat[idx * 86: (idx + 1) * 86] = hist_obs

            # Stand policy when speed is small, else Walk policy
            is_standing = np.linalg.norm(self.loco_cmd) <= 0.05
            session = self.stand_session if is_standing else self.walk_session

            ort_inputs = {session.get_inputs()[0].name: self.obs_flat.reshape(1, 516)}
            ort_outs = session.run(None, ort_inputs)
            self.wbc_action = ort_outs[0].squeeze()

            self.target_dof_pos = self.wbc_action * self.config["action_scale"] + self.config["default_angles"]

        # Calculate PD Control Torques
        self.data.qfrc_applied[:] = 0.0

        current_leg_q = np.array([self.data.qpos[adr] for adr in self.leg_qpos_adrs], dtype=np.float32)
        current_leg_dq = np.array([self.data.qvel[adr] for adr in self.leg_dof_adrs], dtype=np.float32)
        leg_torques = pd_control(self.target_dof_pos, current_leg_q, self.config["kps"], 0.0, current_leg_dq, self.config["kds"])

        current_arm_q = np.array([self.data.qpos[adr] for adr in self.arm_qpos_adrs], dtype=np.float32)
        current_arm_dq = np.array([self.data.qvel[adr] for adr in self.arm_dof_adrs], dtype=np.float32)
        arm_torques = pd_control(0.0, current_arm_q, self.arm_kps, 0.0, current_arm_dq, self.arm_kds)

        current_hand_q = np.array([self.data.qpos[adr] for adr in self.hand_qpos_adrs], dtype=np.float32)
        current_hand_dq = np.array([self.data.qvel[adr] for adr in self.hand_dof_adrs], dtype=np.float32)
        hand_torques = pd_control(0.0, current_hand_q, self.hand_kps, 0.0, current_hand_dq, self.hand_kds)

        for i, dof_adr in enumerate(self.leg_dof_adrs):
            self.data.qfrc_applied[dof_adr] = leg_torques[i]
        for i, dof_adr in enumerate(self.arm_dof_adrs):
            self.data.qfrc_applied[dof_adr] = arm_torques[i]
        for i, dof_adr in enumerate(self.hand_dof_adrs):
            self.data.qfrc_applied[dof_adr] = hand_torques[i]

        # Advance physics simulation
        mujoco.mj_step(self.model, self.data)
        self.step_counter += 1

    def _get_obs(self):
        pos = self.data.xpos[self.pelvis_id, :2]
        yaw = self._get_yaw()
        vel = self.data.qvel[:2]

        goal_vec = self.goal - pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-6)

        lidar = get_lidar_readings(
            self.model, self.data, self.pelvis_id,
            n_rays=N_LIDAR_RAYS, max_range=MAX_LIDAR_RANGE
        )

        obs = np.concatenate([
            pos,                    # 2
            [yaw],                  # 1
            vel,                    # 2
            goal_dir,               # 2
            [goal_dist / 6.0],      # 1 (normalized by room size)
            lidar,                  # N_LIDAR_RAYS
        ]).astype(np.float32)

        return obs

    def _compute_reward(self):
        # ORIGINAL (buggy) reward -- no heading/orientation term at all.
        # Only cares whether Euclidean distance-to-goal decreased, so walking
        # backward toward a goal scores exactly as well as walking forward
        # toward it. This is the version that produced the backward-shuffle-
        # into-a-wall behavior. See module docstring for the fix.
        dist = self._dist_to_goal()

        # Progress reward — did we get closer?
        progress = self._prev_dist_to_goal - dist
        reward = progress * 5.0 #previously was 2.0

        # Goal reached
        if dist < 0.4:
            reward += 50.0 #previously was 10

        # Collision penalty
        if self._is_collision():
            reward -= 5.0

        # Small step penalty — encourages efficiency
        reward -= 0.01

        return float(reward)

    def _dist_to_goal(self):
        pos = self.data.xpos[self.pelvis_id, :2]
        return float(np.linalg.norm(self.goal - pos))

    def _is_done(self):
        return self._dist_to_goal() < 0.4 or self._is_collision()

    def _is_collision(self):
        # Check if robot fell over
        pelvis_z = self.data.xpos[self.pelvis_id, 2]
        if pelvis_z < 0.4:
            return True
        # Check contacts with obstacles - ignore floor
        floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            #Skip floor contacts
            if contact.geom1 == floor_id or contact.geom2 == floor_id:
                continue
            if contact.dist < -0.02:  # significant penetration
                return True
        return False

    def _get_yaw(self):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


if __name__ == "__main__":
    env = G1NavigationEnv()

    # Sanity check
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment OK")

    # Train
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    print("Starting training (ILLUSTRATION RUN -- reproduces backward-walking behavior)...")
    model.learn(total_timesteps=100_000)
    model.save("navigation_policy_backward_walk_example")
    print("Saved to navigation_policy_backward_walk_example.zip")
