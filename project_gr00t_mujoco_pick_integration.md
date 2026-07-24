---
name: project-gr00t-mujoco-pick-integration
description: "Connecting Isaac-GR00T's real policy inference to the g1-with-brainco-hand MuJoCo sim to pick up a cube -- architecture, confirmed checkpoint facts, bugs fixed, and current unresolved stall"
metadata: 
  node_type: memory
  type: project
  originSessionId: f713a406-c514-4aef-9edf-89b5cfad3029
  modified: 2026-07-23T10:59:37.759Z
---

Goal: run the real, trained GR00T N1.7 checkpoint (not mocked) against a MuJoCo
simulation of the G1 + BrainCo-hand robot to pick up a cube. Two separate
projects had to be bridged: [[project-brainco-setup]] covers the training side.

## Environment split (the actual "how do you connect them" answer)

`g1-with-brainco-hand/` (a separate git repo, copied in for convenience) has its
own venv with only `mujoco`+`onnxruntime` -- no gr00t/torch. Isaac-GR00T's main
`.venv` has gr00t+torch but lacked `mujoco`. Fix: `uv add mujoco` into
Isaac-GR00T's own `.venv` (confirmed installed, mujoco==3.10.0) so ONE process
can import both `gr00t.policy.gr00t_policy.Gr00tPolicy` and drive the MuJoCo
sim directly -- no client/server split needed for this local-sim use case.

## Files created this session (all additive -- see [[feedback-no-silent-file-changes]])

- `g1-with-brainco-hand/scenes/mujoco/rubiks_cube_scene.xml` -- new scene
  (reuses the existing `g1_fixed_manipulation.xml` robot include). White table
  (`mat_table`), a plain solid-color cube body `rubiks_cube` (NOT
  rainbow-sticker rubik's-style -- user asked for a plain cube) at
  `pos="0.50 0 0.728"`, a black plate/tray body `plate` at
  `pos="0.65 0 0.706"` (the real task instruction references "the plate"),
  and a custom camera `head_camera_close` (explicit baked pos+xyaxes, not
  `target=`) tuned to match the real training footage's close/steep-down
  framing -- built by iterating with `mujoco.MjvCamera` FREE mode
  (lookat/distance/azimuth/elevation) then extracting the exact resulting
  pos/forward/up vectors via `renderer.scene.camera[0]` and baking those in.
- `g1-with-brainco-hand/src/g1_control/vla_env_gr00t.py` -- new sibling env
  class `G1BrainCoPickEnv`. The ORIGINAL `vla_env.py` (single left arm, apple
  scene, mock inference) was left completely untouched.
- `g1-with-brainco-hand/scripts/gr00t_pick_rubiks_cube.py` -- new driver
  script. Loads `Gr00tPolicy(embodiment_tag="new_embodiment", model_path=outputs/g1_brainco_pick/checkpoint-20000, device="cuda")`
  directly. Receding-horizon loop, re-queries once per 16-step chunk. Always
  saves a video (`outputs/g1_brainco_pick/mujoco_pick_rollout.mp4`); `--viewer`
  additionally opens a live `mujoco.viewer.launch_passive` window.

## Run commands

- Headless (video only): `MUJOCO_GL=egl .venv/bin/python g1-with-brainco-hand/scripts/gr00t_pick_rubiks_cube.py --num-chunks N`
- Live viewer (needs a real display -- do NOT set `MUJOCO_GL=egl` here, it conflicts with the GUI window): `.venv/bin/python g1-with-brainco-hand/scripts/gr00t_pick_rubiks_cube.py --viewer --num-chunks N`
- MuJoCo GUI's built-in "Screenshot" button saves a fixed `screenshot.png` (overwritten each time) to the process's *current working directory* -- confirmed from strings in the compiled `_simulate.cpython-310-x86_64-linux-gnu.so`, not documented anywhere obvious.
- Default `--num-chunks` is 15; real demonstrations don't start moving substantially until frame ~100-200 (chunks 7-13), so short rollouts look static even when working correctly.

## Ground-truth checkpoint facts (confirmed from processor_config.json / statistics.json / the raw HF dataset's own info.json -- NOT from this repo's markdown docs)

**Important:** several existing docs in the main repo (`RUNS_SUMMARY_AND_GUIDE.md`,
`RUN_4_EGO_ONLY_TEMPLATE_README.md`, `outputs/g1_brainco_pick/RUN_4_PICKING_DIVERSITY_README.md`)
contain fabricated/inconsistent numbers -- invented loss curves, claims of 3
cameras/105D state that contradict the checkpoint's actual baked-in config,
and even quadruped joint names (`FL_hip`, `FL_thigh`, `FL_calf`) in a humanoid
doc. Don't trust those at face value; verify against the checkpoint's own
`processor/processor_config.json` and `processor/statistics.json` instead.

- Checkpoint: `outputs/g1_brainco_pick/checkpoint-20000`, tag `new_embodiment`,
  trained on 8 BrainCo picking tasks (`dataset_g1/brainco_pick/*`) including
  `G1_Brainco_GraspRubiksCube_Dataset` (197 episodes).
- Real modality config: video = **1 ego camera only** (`observation.images.head_stereo_left`,
  ego-only, wrist cams dropped); state = `arm_q`(14D)+`hand_state`(12D);
  action = `arm_q`(14D, RELATIVE, decoded to ABSOLUTE automatically by
  `Gr00tPolicy.get_action()`) + `hand_cmd`(12D, ABSOLUTE), 16-step horizon.
- `arm_q` 14D order (confirmed from the raw HF dataset's `info.json` feature
  names): left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
  wrist_roll, wrist_pitch, wrist_yaw), then right arm, same order.
- Hand 12D order: per-hand `[Thumb, ThumbAux, Index, Middle, Ring, Pinky]`,
  left then right. `hand_state`/`hand_cmd` are normalized **[0,1] closure
  fractions per finger, NOT raw radians** (confirmed from statistics.json --
  real joint ranges go up to ~1.7 rad while trained values live in [0,1]).
  Whether "Thumb" = rotation-first or flex-first vs. our grouping's assumed
  order is still unconfirmed -- residual uncertainty, low priority since the
  arms are the main mobility for this task.
- `Gr00tPolicy.get_action()` returns the action dict keyed by the **full
  modality name with the "action." prefix** (`"action.arm_q"`,
  `"action.hand_cmd"`), not the bare name -- easy first bug to hit.
- Real training instruction text (exact, from `meta/tasks.jsonl`): *"Pick up
  the Rubik's Cube and put it in the plate"*. Using this exact phrasing vs. a
  generic paraphrase measurably changed the model's predicted hand-closure
  deltas in a side-by-side check -- phrasing matters here.

## Sim-side bugs found and fixed (each gave a real, confirmed improvement)

1. Blanket `±3.2` rad clip on all 14 arm joints (instead of each joint's own
   physical range) let the model push joints into physically-impossible
   poses -- e.g. `right_shoulder_roll`'s real range is `[-2.2515, 1.5882]`,
   nearly a full radian narrower than the blanket clip on one side. Fixed:
   clip each joint to its own compiled `jnt_range`.
2. Waist joints (`waist_yaw/roll/pitch`) are free/undamped and outside GR00T's
   action space -- nothing held them, so gravity sagged/rotated the torso
   (and the head camera bolted to it) within ~1-2s, dragging the camera
   completely off the scene. Fixed: locked to 0 every step.
3. No actuators exist anywhere in this robot model. Setting a joint's `qpos`
   once per 16-step-hold and letting physics free-run for the rest caused
   visible "buzzing" on the light, undamped BrainCo finger joints touching
   the cube (and was likely the cause of recurring "NaN/Inf in QACC"
   instability warnings). Fixed: re-apply `qpos` AND zero `qvel` every single
   physics substep, not just once per hold -- approximates an infinitely
   stiff position controller.
4. Default reset pose was guessed (elbow bent 0.3 rad, everything else 0)
   instead of matching real data. Since `arm_q` is RELATIVE (decoded against
   current state), starting far outside the training distribution made the
   model's very first prediction in a rollout unreliable. Fixed: pulled
   `default_arm_q`/`default_hand_frac` directly from
   `G1_Brainco_GraspRubiksCube_Dataset` episode_000001 frame 0.
5. Camera framing was wrong -- the shared `head_camera` (defined in the
   pre-existing `g1_fixed_manipulation.xml`, not touched) is a generic robot
   camera with only a ~30° downward tilt, so the sim's view was dominated by
   floor/background with the table a tiny wedge and the cube barely visible.
   Real footage is close and steep (~65-70° down), table+hands filling almost
   the whole frame. Fixed: added `head_camera_close`, a new world-fixed
   camera in the new scene file (fine since the waist is now locked static
   anyway), tuned to match.

## Current status: pipeline works end-to-end but the grasp doesn't complete

Stable (no NaN/instability, no camera drift, no buzzing), checkpoint loads,
produces real chunk-by-chunk predictions. Measured directly with MuJoCo's own
contact list and body-distance tracking over 60 chunks (960 steps -- well past
the real demo's active-reach window of frames ~100-800):

- **Right hand never contacts the cube.** Closest approach: `right_index_tip`
  gets to **0.1847-0.1855m** from the cube center (contact needs ~0.03-0.05m,
  given the cube's 0.028m half-size) around chunk 10, then **reverses and
  drifts away** instead of continuing to close the gap -- ends up 3x farther
  than the start by chunk 60 without mitigation, ~2x farther with it.
- Tried: rate-limiting `arm_q` deltas per step (`max_arm_delta=0.06` rad,
  in `G1BrainCoPickEnv.step()`) to damp a suspected integrator-windup-like
  loop (relative-action decoding referencing an already-clipped, never-seen-
  in-training state, causing the model to keep pushing the same direction
  each chunk instead of correcting). Confirmed via measurement: the real
  episode's own joints NEVER approach their hardware limits anywhere in 1801
  frames, so hitting a limit in our sim is definitely not reproducing genuine
  policy behavior. The rate limit reduced the SEVERITY of the post-stall
  drift (final distance 0.56m vs. 0.79m over 960 steps) but did **not**
  change the ~18.5cm stall point itself -- so windup is a real contributing
  factor but not the root cause of why it can't close the last ~15cm.

## Suggested next steps (not yet tried)

- Sweep the cube's table position -- it was placed somewhat arbitrarily
  (`0.50 0 0.728`), never validated against the training distribution of
  object placements. Check whether ~18cm is a placement-specific ceiling.
- Verify `head_camera_close`'s pose more rigorously against the real robot's
  actual physical camera calibration -- current tuning was by visual
  comparison only, not exact intrinsics/extrinsics replication.
- Resolve the thumb rotation-vs-flex ordering ambiguity in `HAND_GROUPS`.
- User's idea (approved concept, not yet implemented): detect an anomalous
  chunk (would require clipping to a hard joint limit, or deltas far outside
  real per-chunk statistics) and **reject-and-hold** (skip applying that
  chunk's motion, re-query fresh next chunk) rather than resetting all the
  way to the default pose -- a full reset risks looping identically forever
  since the image wouldn't meaningfully change. Caveat: leans on privileged
  sim-only ground truth (exact joint limits) as the detector; a real
  deployment would need its own anomaly/safety monitor.

## How to apply

Read `g1-with-brainco-hand/src/g1_control/vla_env_gr00t.py` and
`g1-with-brainco-hand/scripts/gr00t_pick_rubiks_cube.py` for current state
before making changes -- this memory may drift from the code. Re-verify the
"current status" numbers by re-running the diagnostic pattern (contact-list
check + body-distance tracking over ~60 chunks) rather than assuming the
stall point is still exactly 18.5cm after further changes.
