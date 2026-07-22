import os
import sys
import datetime
import numpy as np
import mujoco
import mujoco.viewer
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from stable_baselines3 import PPO
from train_g1_navigate import G1NavigationEnv

# Success/miss threshold -- matches the env's own goal-reached check in _compute_reward.
SUCCESS_DIST_THRESHOLD = 0.4

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOG_PATH = os.path.join(ROOT_DIR, 'navigation_eval.log')
log_file = open(LOG_PATH, 'a')

def log_line(text):
    """Print to console and append to navigation_eval.log (flushed immediately)."""
    print(text)
    log_file.write(text + "\n")
    log_file.flush()

# Load trained policy
model = PPO.load("navigation_policy")

# Create environment
env = G1NavigationEnv()
obs, _ = env.reset()

run_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_line("=" * 80)
log_line(f"Run started: {run_start}")
log_line("=" * 80)

log_line("=== Navigation Policy Evaluation ===")
log_line(f"Goal: {env.goal}")

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
            dist = float(np.linalg.norm(env.goal - pos))
            status = "SUCCESS" if dist < SUCCESS_DIST_THRESHOLD else "MISS"
            log_line(f"Episode {episode}: goal={env.goal.tolist()} dist={dist:.3f} -> {status}")
            obs, _ = env.reset()
            log_line(f"New goal: {env.goal}")