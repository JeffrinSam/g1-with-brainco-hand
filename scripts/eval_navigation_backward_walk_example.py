"""
eval_navigation_backward_walk_example.py

ILLUSTRATION FILE — companion to train_g1_navigate_backward_walk_example.py.
Same as eval_navigation.py, but loads navigation_policy_backward_walk_example.zip
and imports G1NavigationEnv from the illustration training file (the one with
the original, no-heading-term reward that produces backward-walking-into-walls
behavior). Run after training the illustration policy, to watch the bug live.
"""

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from stable_baselines3 import PPO
from train_g1_navigate_backward_walk_example import G1NavigationEnv

# Load trained policy
model = PPO.load("navigation_policy_backward_walk_example")

# Create environment
env = G1NavigationEnv()
obs, _ = env.reset()

print("=== Navigation Policy Evaluation (backward-walk illustration) ===")
print(f"Goal: {env.goal}")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.distance = 8.0
    viewer.cam.lookat = np.array([2.0, 0.0, 0.5])

    episode = 0
    while viewer.is_running():
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        viewer.sync()
        time.sleep(0.02) #previously 0.002 - was to fast, movements too edgy

        if terminated or truncated:
            episode += 1
            pos = env.data.xpos[env.pelvis_id, :2]
            print(f"Episode {episode} ended | pos: {pos} | goal: {env.goal}")
            obs, _ = env.reset()
            print(f"New goal: {env.goal}")
