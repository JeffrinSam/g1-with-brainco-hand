import mujoco

# 1. Create a URDF without top-level material tags
urdf_content = """<robot name="g1">
  <link name="world"/>
  <joint name="floating_base" type="floating">
    <parent link="world"/>
    <child link="pelvis"/>
  </joint>
  <link name="pelvis">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="3.8"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.2"/></geometry>
      <material name="dark"/>
    </visual>
  </link>
</robot>"""

with open("D:/Personal/Mujcotutorial/g1_with_brainco_hand/test_robot.urdf", "w") as f:
    f.write(urdf_content)

# 2. Create a scene XML that defines the assets and includes the URDF
scene_xml = """<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <texture type="2d" name="grid" builtin="checker" width="100" height="100" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"/>
    <material name="dark" texture="grid" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="dark"/>
    <include file="test_robot.urdf"/>
  </worldbody>
</mujoco>"""

with open("D:/Personal/Mujcotutorial/g1_with_brainco_hand/test_scene.xml", "w") as f:
    f.write(scene_xml)

try:
    m = mujoco.MjModel.from_xml_path("D:/Personal/Mujcotutorial/g1_with_brainco_hand/test_scene.xml")
    print("SUCCESS! Materials:", m.nmat, "Textures:", m.ntex, "Geoms:", m.ngeom)
except Exception as e:
    print("FAILED:", str(e))
