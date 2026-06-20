"""
run_simulation.py

A self-contained script to run a headless physical simulation of the Unitree G1 
robot with BrainCo Dexterous Hands in MuJoCo 3.3.3+.
This script demonstrates:
1. Loading the compiled g1_fixed.urdf.
2. Initializing MjModel and MjData.
3. Simulating 1.0 second of simulation time headlessly.
4. Logging key joint states at 0.1-second intervals.
"""

import os
import argparse
import mujoco
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Headless MuJoCo Simulation for Unitree G1 with BrainCo Hands")
    parser.add_argument("--duration", type=float, default=1.0, help="Simulation duration in seconds")
    parser.add_argument("--timestep", type=float, default=0.002, help="Physics simulation timestep")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "g1_fixed.urdf")

    print("=== Principal MuJoCo Simulation: Headless G1 Run ===")
    print(f"Loading fixed URDF: {urdf_path}")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"Model file not found at {urdf_path}")

    # 1. Model & Data separation
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)

    # Disable gravity by default to keep the fixed/floating robot standing without collapse
    model.opt.gravity = (0.0, 0.0, 0.0)

    # Set timestep if specified
    model.opt.timestep = args.timestep
    print(f"Model successfully compiled.")
    print(f"Bodies count: {model.nbody}")
    print(f"Degrees of freedom (nv): {model.nv}")
    print(f"Actuators count (nu): {model.nu}")
    print(f"Timestep: {model.opt.timestep} seconds")

    # Let's inspect joint names
    joint_names = []
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(jname)
    print(f"First 10 joints: {joint_names[:10]} ...")

    # 2. Simulation loop
    sim_duration = args.duration
    total_steps = int(sim_duration / model.opt.timestep)
    print_interval = 0.1
    steps_between_prints = int(print_interval / model.opt.timestep)

    print(f"\nRunning simulation for {sim_duration}s of virtual time...")
    print(f"{'Time (s)':<10} | {'Torso Z pos':<12} | {'Waist Yaw Angle (rad)':<22} | {'Left Index Proximal (rad)':<25}")
    print("-" * 80)

    # Let's find index positions of specific joints and bodies
    waist_yaw_id = -1
    left_index_prox_id = -1

    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname == "waist_yaw_joint":
            waist_yaw_id = i
        elif jname == "left_index_proximal_joint":
            left_index_prox_id = i

    # Find torso body ID for Cartesian position tracking
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    def get_joint_qpos_val(j_id):
        if j_id != -1:
            qpos_adr = model.jnt_qposadr[j_id]
            return data.qpos[qpos_adr]
        return 0.0

    trajectory_log = []

    for step in range(total_steps):
        # Step physics forward
        mujoco.mj_step(model, data)

        # Apply basic damping or small gravity compensating control targets (PD-like)
        # G1 has actuators, data.ctrl can be set to test movement
        # Set small test command to waist yaw joint to show activity
        # Let's find the actuator ID for waist yaw
        if step == 0:
            # list some actuator names
            actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
            print(f"Exposed Actuators (first 5): {actuator_names[:5]}")

        # At each interval, print stats
        if step % steps_between_prints == 0:
            # Get cartesian position of torso body
            torso_z = data.xpos[torso_body_id, 2] if torso_body_id != -1 else 0.0
            waist_yaw_val = get_joint_qpos_val(waist_yaw_id)
            left_index_prox_val = get_joint_qpos_val(left_index_prox_id)
            
            print(f"{data.time:<10.3f} | {torso_z:<12.5f} | {waist_yaw_val:<22.5f} | {left_index_prox_val:<25.5f}")
            trajectory_log.append((data.time, torso_z, waist_yaw_val, left_index_prox_val))

    print("-" * 80)
    print("Simulation run completed successfully.")
    
    # Save log to file
    log_path = os.path.join(script_dir, "simulation_run.log")
    with open(log_path, "w") as f:
        f.write("Time,Torso_Z,Waist_Yaw,Left_Index_Proximal\n")
        for entry in trajectory_log:
            f.write(f"{entry[0]},{entry[1]},{entry[2]},{entry[3]}\n")
    print(f"Log file saved to: {log_path}")

if __name__ == "__main__":
    main()
