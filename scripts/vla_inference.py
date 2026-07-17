"""
vla_inference.py - GR00T-G1 VLA Inference Loop

Runs a 5-step control loop against G1VLAEnv (src/g1_control/vla_env.py):
fetch a head-camera frame + left-arm joint state, run VLA inference, step
the environment with the predicted left-arm action. No viewer — uses
G1VLAEnv's offscreen renderer, so a plain `python` works (no mjpython needed).

USE_ACTUAL_MODEL (below) is hardcoded False: real GR00T-N1.7 inference is
wired up but not runnable without the `huggingface_hub`/`gr00t` packages and
model access, so this currently runs in mock mode (random small action deltas).

Usage Examples:
----------------
1. Run the 5-step mock inference loop:
       python scripts/vla_inference.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "g1_control"))
from vla_env import G1VLAEnv

# Flag to easily switch between actual model loading and mock inference testing
USE_ACTUAL_MODEL = False

if USE_ACTUAL_MODEL:
    from huggingface_hub import snapshot_download
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    
    # Download weights
    print("Downloading GR00T-N1.7 checkpoint...")
    model_path = snapshot_download("JeffrinSam/GR00T-N1.7-G1-BrainCo-Pick")
    
    # Initialize pipeline
    pipeline = Gr00tN1d7Pipeline.from_pretrained(
        "JeffrinSam/GR00T-N1.7-G1-BrainCo-Pick",
        modality_config_path="modality_config.py",
    )
else:
    pipeline = None
    print("[Mode] Running with MOCK VLA Model (USE_ACTUAL_MODEL = False).")

def run_vla_inference(env_observation, instruction):
    """
    Formulates observations into VLA inputs, runs the pipeline, and returns 
    the predicted action outputs.
    """
    # 1. Fetch camera frame (H, W, 3) uint8
    head_frame = env_observation["image"]
    
    # 2. Map 7D arm states to 14D inputs (Left arm + stationary Right arm)
    left_arm_q = env_observation["qpos"]
    right_arm_q = np.array([0.0, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0]) # Stationary default
    arm_q = np.concatenate([left_arm_q, right_arm_q])            # 14D position vector
    
    # 3. Map hand state to 12D inputs (Left hand + Right hand)
    left_hand_state = np.zeros(6)  # Mocked finger state
    right_hand_state = np.zeros(6)
    hand_state = np.concatenate([left_hand_state, right_hand_state])
    
    if USE_ACTUAL_MODEL and pipeline is not None:
        # Query NVIDIA GR00T-N1.7 model
        output = pipeline.get_action(
            observation={
                "observation.images.head_stereo_left": head_frame,
                "observation.state.arm_q": arm_q,
                "observation.state.hand_state": hand_state,
            },
            language_instruction=instruction,
        )
        # Returns action horizons
        arm_actions = output["action.arm_q"]     # Shape (16, 14)
        hand_commands = output["action.hand_cmd"] # Shape (16, 12)
        
        # We execute the first action in the 16-step predicted horizon
        # returning only the left arm joints delta values [0:7]
        return arm_actions[0][:7]
    else:
        # Mock action prediction matching output shapes
        print(f"[VLA Inference] Processing image size {head_frame.shape} with prompt: '{instruction}'")
        return np.random.uniform(-0.01, 0.01, size=7)

def main():
    """Run a 5-step VLA control loop: reset the env, then repeatedly infer and apply an action."""
    env = G1VLAEnv()
    obs = env.reset()
    
    instruction = "Pick up the apple"
    print(f"\nInitialized VLA inference. Target: '{instruction}'")
    
    # Execute loop
    for step in range(5):
        print(f"\n--- Control Step {step} ---")
        action = run_vla_inference(obs, instruction)
        obs = env.step(action)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
