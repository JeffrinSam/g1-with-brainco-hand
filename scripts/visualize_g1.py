"""
visualize_g1.py

Reads assets/robots/g1_29dof_mode_15_brainco_hand.urdf, clears its
meshdir="meshes" attribute (so mesh paths resolve relative to the URDF's own
folder), writes the result to assets/robots/g1_fixed.urdf, then loads and
visualizes that fixed URDF in the interactive passive viewer. Gravity is off
by default so the robot doesn't collapse; the base is fixed to the world
unless --free-base is passed.

Usage Examples:
----------------
1. Inspect the robot standing still, gravity off, fixed base (macOS needs
   `mjpython` in place of `python`):
       mjpython scripts/visualize_g1.py

2. Let it fall under gravity with a free-floating base:
       mjpython scripts/visualize_g1.py --gravity --free-base
"""

import os
import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np

def main():
    """Patch the raw URDF's meshdir, write it as g1_fixed.urdf, then load and open the passive viewer."""
    parser = argparse.ArgumentParser(description="Visualize G1 with BrainCo Hands in MuJoCo")
    parser.add_argument("--gravity", action="store_true", help="Enable gravity (disabled by default to prevent collapse)")
    parser.add_argument("--free-base", action="store_true", help="Enable free-floating base (otherwise fixed to world)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    robots_dir = os.path.join(root_dir, "assets", "robots")
    urdf_path = os.path.join(robots_dir, "g1_29dof_mode_15_brainco_hand.urdf")
    fixed_urdf_path = os.path.join(robots_dir, "g1_fixed.urdf")

    # 1. Read URDF and resolve meshdir paths
    print(f"Reading URDF from: {urdf_path}")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_content = f.read()

    # Clear meshdir="meshes" attribute to resolve mesh paths relative to URDF folder
    fixed_content = urdf_content.replace('meshdir="meshes"', 'meshdir=""')

    # If free-base is requested, uncomment the floating base joint
    if args.free_base:
        print("Enabling free-floating base joint...")
        fixed_content = fixed_content.replace('<!-- <link name="world"></link>', '<link name="world"></link>')
        fixed_content = fixed_content.replace('<parent link="world"/>\n    <child link="pelvis"/>\n  </joint> -->', '<parent link="world"/>\n    <child link="pelvis"/>\n  </joint>')

    with open(fixed_urdf_path, "w", encoding="utf-8") as f:
        f.write(fixed_content)
    print(f"Saved fixed URDF to: {fixed_urdf_path}")

    # 2. Compile model
    model = mujoco.MjModel.from_xml_path(fixed_urdf_path)
    data = mujoco.MjData(model)

    # Disable gravity by default to keep the fixed/floating robot standing for inspection
    if not args.gravity:
        print("Gravity disabled (run with --gravity to enable)")
        model.opt.gravity = (0.0, 0.0, 0.0)
    else:
        print("Gravity enabled")

    # 3. Print joint information for inspection
    print(f"\n=== Model Information ===")
    print(f"Generalized Coordinates (nq): {model.nq}")
    print(f"Degrees of Freedom (nv): {model.nv}")
    print(f"Actuators (nu): {model.nu}")
    print(f"Bodies: {model.nbody}")

    print("\nJoints present in the model:")
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i]
        jrange = model.jnt_range[i]
        type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, "unknown")
        print(f" - Joint {i:02d}: '{jname}' (type: {type_str}, range: {jrange})")

    # 4. Launch interactive passive viewer
    print("\nLaunching MuJoCo passive viewer...")
    print("-> Hold [Left Click + Drag] to rotate the camera.")
    print("-> Hold [Right Click + Drag] to pan.")
    print("-> Double-click a body or joint, then hold [Ctrl + Left Click + Drag] to apply forces/torques.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera position for a nice front view
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.distance = 2.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.5])

        prev_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            
            # Step the physics
            mujoco.mj_step(model, data)
            
            # Sync physics state to visualizer
            viewer.sync()

            # Maintain physics time-step
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    print("Visualization closed.")

if __name__ == "__main__":
    main()
