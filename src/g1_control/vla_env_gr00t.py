"""
vla_env_gr00t.py - Bimanual G1 + BrainCo VLA Environment for real GR00T N1.7 inference

Sibling of vla_env.py (which stays as-is: single left arm, apple scene, mock
inference). This env is shaped specifically to match the real, trained
checkpoint at Isaac-GR00T/outputs/g1_brainco_pick/checkpoint-20000, whose
modality config (baked into its processor_config.json; also mirrored in
examples/G1_Brainco/g1_brainco_config_pick_ego.py in the main repo) is:
  - video:    1 ego head camera   -> "observation.images.head_stereo_left"
  - state:    "observation.state.arm_q" (14D bimanual, left-then-right) +
              "observation.state.hand_state" (12D bimanual BrainCo, left-then-right)
  - language: single task-instruction string
  - action:   "arm_q" (14D) + "hand_cmd" (12D), 16-step horizon -- both already
              decoded to absolute values by Gr00tPolicy.get_action(), so
              step() applies them directly (no delta integration needed).

hand_state/hand_cmd are trained normalized to a [0, 1] closure fraction per
finger (0=open, 1=fully closed) -- confirmed from checkpoint statistics.json,
where hand values live in [0, 1] while the sim's own joint ranges go up to
~1.7 rad. So this env converts each finger's raw joint radians to/from that
[0, 1] fraction using the joint's own compiled range.

Loads scenes/mujoco/rubiks_cube_scene.xml. Used by
scripts/gr00t_pick_rubiks_cube.py; not a standalone script.
"""

import os
import time
import numpy as np
import mujoco


LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# 14D order matching this repo's other bimanual joint lists (run_wbc_mujoco.py): left arm, then right arm.
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# The pelvis itself has no joint (rigidly fixed to the world in
# g1_fixed_manipulation.xml), but these 3 waist joints above it are free and
# undamped -- outside GR00T's action space, so nothing holds them, and gravity
# sags/rotates the torso (and the head camera bolted to it) within ~1-2s of
# sim time if left alone. Locked to 0 every step, same as the arm/hand joints.
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


def _hand_joint_groups(prefix):
    """6 actuated-DOF groups per BrainCo hand: thumb rotation + one flex value
    per digit. Each flex value drives both the proximal and distal joint of
    that finger (approximates the real hand's tendon coupling -- the sim URDF
    exposes them as independent joints)."""
    return [
        [f"{prefix}_thumb_metacarpal_joint"],
        [f"{prefix}_thumb_proximal_joint", f"{prefix}_thumb_distal_joint"],
        [f"{prefix}_index_proximal_joint", f"{prefix}_index_distal_joint"],
        [f"{prefix}_middle_proximal_joint", f"{prefix}_middle_distal_joint"],
        [f"{prefix}_ring_proximal_joint", f"{prefix}_ring_distal_joint"],
        [f"{prefix}_pinky_proximal_joint", f"{prefix}_pinky_distal_joint"],
    ]


# 12 groups -> hand_state/hand_cmd order: left hand (6), then right hand (6).
HAND_GROUPS = _hand_joint_groups("left") + _hand_joint_groups("right")


class G1BrainCoPickEnv:
    """VLA-facing wrapper around the rubik's-cube/table MuJoCo scene: ego camera
    + bimanual arm/hand state in, bimanual arm/hand absolute targets out."""

    def __init__(self, scene_xml=None, camera="head_camera_close"):
        """Load the scene, set up the head-camera offscreen renderer, and resolve joint addresses."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(os.path.dirname(self.script_dir))
        self.scene_xml = scene_xml or os.path.join(
            self.root_dir, "scenes", "mujoco", "rubiks_cube_scene.xml"
        )

        if not os.path.exists(self.scene_xml):
            raise FileNotFoundError(f"Scene XML not found at {self.scene_xml}")

        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        self.width = 640
        self.height = 480
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.camera = camera

        self.arm_qpos_adrs = [self._joint_qpos_adr(name) for name in ARM_JOINTS]
        self.arm_dof_adrs = [self._joint_dof_adr(name) for name in ARM_JOINTS]
        self.arm_ranges = [self._joint_range(name) for name in ARM_JOINTS]
        self.hand_groups = [
            [
                (self._joint_qpos_adr(name), self._joint_dof_adr(name), *self._joint_range(name))
                for name in group
            ]
            for group in HAND_GROUPS
        ]
        self.cube_qpos_adr = self._joint_qpos_adr("cube_free")
        self.waist_qpos_adrs = [self._joint_qpos_adr(name) for name in WAIST_JOINTS]
        self.waist_dof_adrs = [self._joint_dof_adr(name) for name in WAIST_JOINTS]

        # Reset pose taken directly from a real training episode's frame 0
        # (dataset_g1/brainco_pick/G1_Brainco_GraspRubiksCube_Dataset,
        # episode_000001 observation.state) rather than guessed -- arm_q is a
        # RELATIVE action decoded against the current state, so the model's
        # very first prediction in a rollout is conditioned on this pose, and
        # starting far outside the training distribution (an all-zeros pose)
        # made its first predictions unreliable.
        self.default_arm_q = np.array([
            -0.839, 0.145, 0.235, 1.298, -0.042, -0.351, -0.141,   # left arm
            -0.773, -0.184, -0.117, 1.314, -0.137, -0.533, -0.201,  # right arm
        ], dtype=np.float32)
        # Same episode/frame, hand_state (already the [0, 1] closure fraction
        # this env's hand_groups use -- positional order matches HAND_GROUPS).
        self.default_hand_frac = np.array([
            0.469, 0.693, 0.184, 0.274, 0.269, 0.284,  # left hand
            0.106, 0.628, 0.271, 0.351, 0.360, 0.367,  # right hand
        ], dtype=np.float32)

    def _joint_qpos_adr(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint '{name}' not found in {self.scene_xml}")
        return self.model.jnt_qposadr[jid]

    def _joint_dof_adr(self, name):
        # NOT the same as the qpos address once a free joint (7 qpos, 6 dof --
        # quaternion vs. angular velocity) precedes this one in the model, as
        # the cube's free joint does here.
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_dofadr[jid]

    def _joint_range(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = self.model.jnt_range[jid]
        if lo == hi:  # unlimited/fixed joints report a zero-span range
            return -1e6, 1e6
        return float(lo), float(hi)

    def reset(self, cube_xy=None):
        """Reset physics, set a reaching-ready arm pose, (re)place the cube, and return the initial observation."""
        mujoco.mj_resetData(self.model, self.data)

        for adr, q in zip(self.arm_qpos_adrs, self.default_arm_q):
            self.data.qpos[adr] = q

        for adr in self.waist_qpos_adrs:
            self.data.qpos[adr] = 0.0

        for group, frac in zip(self.hand_groups, self.default_hand_frac):
            for adr, _dof_adr, lo, hi in group:
                self.data.qpos[adr] = lo + frac * (hi - lo)

        if cube_xy is not None:
            self.data.qpos[self.cube_qpos_adr] = cube_xy[0]
            self.data.qpos[self.cube_qpos_adr + 1] = cube_xy[1]

        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def cube_height(self):
        """World-frame Z of the cube's free joint (qpos[2] of that joint) -- rises once grasped and lifted."""
        return float(self.data.qpos[self.cube_qpos_adr + 2])

    def _read_arm_q(self):
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adrs], dtype=np.float32)

    def _read_hand_state(self):
        # hand_state is trained normalized to [0, 1] per finger (0=open, 1=fully
        # closed), not raw radians -- report the primary joint of each group
        # (thumb rotation, or each finger's proximal joint) as a fraction of
        # its own [lo, hi] compiled range.
        state = np.empty(len(self.hand_groups), dtype=np.float32)
        for i, group in enumerate(self.hand_groups):
            adr, _dof_adr, lo, hi = group[0]
            state[i] = (self.data.qpos[adr] - lo) / (hi - lo)
        return state

    def get_observation(self, instruction="Pick up the cube"):
        """Return a Gr00tPolicy-shaped observation: nested video/state/language dicts, batch size 1."""
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb_image = self.renderer.render()  # (H, W, 3) uint8

        arm_q = self._read_arm_q()
        hand_state = self._read_hand_state()

        return {
            "video": {
                "observation.images.head_stereo_left": rgb_image[None, None].astype(np.uint8),
            },
            "state": {
                "observation.state.arm_q": arm_q[None, None],
                "observation.state.hand_state": hand_state[None, None],
            },
            "language": {
                "annotation.human.task_description": [[instruction]],
            },
        }

    def step(self, arm_q_target, hand_cmd_target, n_substeps=17, viewer=None, max_arm_delta=0.06):
        """Drive the sim kinematically to the given absolute 14D arm target and
        12D hand closure-fraction target, holding them for n_substeps physics
        steps (~1/30s at this model's 0.002s timestep, matching the ~30fps
        control rate the policy was trained on) before returning the
        resulting observation.

        There are no actuators anywhere in this model, so a target set only
        once and then left alone would immediately start drifting under
        gravity/contact for the rest of the hold (most visible on the
        near-massless, undamped BrainCo finger joints touching the cube).
        Instead, every one of the n_substeps physics steps below re-applies
        qpos AND zeroes qvel for every kinematically-driven joint, which
        approximates an infinitely-stiff position hold: each joint is only
        ever allowed to drift for a single 0.002s physics step before being
        pinned back.

        max_arm_delta rate-limits each arm joint's target to within this many
        radians of its CURRENT actual position (in addition to the hard
        [lo, hi] physical clip below). Without it, a single out-of-range
        prediction gets clipped straight to the joint limit in one step; the
        next chunk's relative-action decode then starts from that abrupt,
        never-seen-in-training state and can keep pushing further into the
        limit instead of correcting (integrator-windup-like failure -- see
        the real episode's own data, where no joint ever comes close to its
        limit, confirming this doesn't happen in genuine demonstrations)."""
        targets = []  # list of (qpos_adr, dof_adr, value)
        for adr, dof_adr, q, (lo, hi) in zip(
            self.arm_qpos_adrs, self.arm_dof_adrs, arm_q_target, self.arm_ranges
        ):
            current_q = self.data.qpos[adr]
            q = np.clip(q, current_q - max_arm_delta, current_q + max_arm_delta)
            targets.append((adr, dof_adr, np.clip(q, lo, hi)))

        for adr, dof_adr in zip(self.waist_qpos_adrs, self.waist_dof_adrs):
            targets.append((adr, dof_adr, 0.0))

        for group, cmd in zip(self.hand_groups, hand_cmd_target):
            frac = np.clip(cmd, 0.0, 1.0)  # hand_cmd is a [0, 1] closure fraction, not radians
            for adr, dof_adr, lo, hi in group:
                targets.append((adr, dof_adr, lo + frac * (hi - lo)))

        for _ in range(n_substeps):
            for adr, dof_adr, value in targets:
                self.data.qpos[adr] = value
                self.data.qvel[dof_adr] = 0.0
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)  # roughly real-time playback

        return self.get_observation()
