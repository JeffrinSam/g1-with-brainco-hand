# Unitree G1 with BrainCo Dexterous Hand (URDF Model)

This repository contains the Unified Robot Description Format (URDF) model and 3D visual/collision meshes for the **Unitree G1 humanoid robot** equipped with the **BrainCo Dexterous Hands**.

This description can be used for simulation, visualization (RViz), and kinematic/dynamic calculations (e.g., in MuJoCo, PyBullet, or ROS).

---

## 🤖 Robot Specifications

* **Robot Name**: `g1_29dof_mode_15_brainco_hand`
* **Total Base Degrees of Freedom (DoF)**: 29 Active DoF (excluding hands)
* **Hand Type**: Left and Right BrainCo Dexterous Hands (5 articulated fingers per hand)

---

## 📂 Repository Structure

```text
g1_with_brainco_hand/
├── README.md                              # This detailed documentation
├── g1_29dof_mode_15_brainco_hand.urdf     # Main Unified Robot Description Format (URDF) file
└── meshes/                                # Visual and collision STL files
    ├── pelvis_ver0529.STL
    ├── head_link.STL
    ├── left_hip_pitch_link.STL
    ├── ... (other link meshes)
    └── right_rubber_hand.STL
```

---

## ⚙️ Joint Configuration Details

### 1. Torso & Waist
* `waist_yaw_joint` (Revolute)
* `waist_roll_joint` (Revolute)
* `waist_pitch_joint` (Revolute)

### 2. Legs (Left & Right)
* `[left/right]_hip_pitch_joint` (Revolute)
* `[left/right]_hip_roll_joint` (Revolute)
* `[left/right]_hip_yaw_joint` (Revolute)
* `[left/right]_knee_joint` (Revolute)
* `[left/right]_ankle_pitch_joint` (Revolute)
* `[left/right]_ankle_roll_joint` (Revolute)

### 3. Arms (Left & Right)
* `[left/right]_shoulder_pitch_joint` (Revolute)
* `[left/right]_shoulder_roll_joint` (Revolute)
* `[left/right]_shoulder_yaw_joint` (Revolute)
* `[left/right]_elbow_joint` (Revolute)
* `[left/right]_wrist_roll_joint` (Revolute)
* `[left/right]_wrist_pitch_joint` (Revolute)
* `[left/right]_wrist_yaw_joint` (Revolute)

### 4. Dexterous Hands (BrainCo)
Each hand features five fully-articulated fingers with high-resolution collision/visual meshes:
* **Thumb**: Metacarpal, Proximal, Distal, and Tip joints.
* **Index**: Proximal, Distal, and Tip joints.
* **Middle**: Proximal, Distal, and Tip joints.
* **Ring**: Proximal, Distal, and Tip joints.
* **Pinky**: Proximal, Distal, and Tip joints.

### 5. Sensors & Auxiliary (Fixed Joints)
* `head_joint` (fixed)
* `imu_in_torso_joint` (fixed)
* `imu_in_pelvis_joint` (fixed)
* `d435_joint` (Intel RealSense D435 camera mount)
* `mid360_joint` (Livox Mid-360 LiDAR mount)

---

## 🚀 Usage

### Relative Paths & ROS/ROS 2
The URDF file references meshes using relative paths:
```xml
<mesh filename="meshes/pelvis_ver0529.STL"/>
```
* **For Direct Use (e.g. PyBullet / MuJoCo)**: The paths will resolve automatically as long as the relative directory structure is preserved.
* **For ROS/ROS 2 package systems**: You may want to prefix the paths with `package://<your_package_name>/` (e.g., `package://g1_description/meshes/...`) depending on your workspace setup.

### MuJoCo Integration
The model includes preconfigured settings for MuJoCo compiler compatibility:
```xml
<mujoco>
  <compiler meshdir="meshes" discardvisual="false" balanceinertia="true"/>
</mujoco>
```

---

*Extracted and organized from the official [unitree_ros](https://github.com/unitreerobotics/unitree_ros) repository.*
