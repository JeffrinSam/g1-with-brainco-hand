"""
gr00t_pick_rubiks_cube.py - Real GR00T N1.7 inference driving the G1+BrainCo
MuJoCo sim (scenes/mujoco/rubiks_cube_scene.xml) to pick up a cube.

This is where the two projects in this repo actually connect:
  - g1-with-brainco-hand/  supplies the MuJoCo scene, robot XML, and
    G1BrainCoPickEnv (src/g1_control/vla_env_gr00t.py) -- pure sim assets, no GR00T deps.
  - Isaac-GR00T/           supplies gr00t.policy.gr00t_policy.Gr00tPolicy,
    which loads outputs/g1_brainco_pick/checkpoint-20000 -- fine-tuned on 8
    BrainCo picking tasks including G1_Brainco_GraspRubiksCube_Dataset (see
    examples/G1_Brainco/g1_brainco_config_pick_ego.py for the exact modality
    config baked into that checkpoint: 1 ego camera, 14D arm_q + 12D
    hand_state/hand_cmd, 16-step action horizon).

Both must run in the SAME Python process, because Gr00tPolicy needs
torch+transformers+gr00t, which g1-with-brainco-hand's own venv doesn't have.
Run this with the Isaac-GR00T repo's own interpreter (mujoco was added there
alongside gr00t/torch) rather than g1-with-brainco-hand/.venv:

Usage Examples:
----------------
1. Run a 3-chunk (48-step) rollout and save it as a video:
       .venv/bin/python g1-with-brainco-hand/scripts/gr00t_pick_rubiks_cube.py

2. Longer rollout against a different checkpoint:
       .venv/bin/python g1-with-brainco-hand/scripts/gr00t_pick_rubiks_cube.py \
           --model-path outputs/g1_brainco_pick/checkpoint-20000 --num-chunks 6
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)  # so `import gr00t` resolves to Isaac-GR00T/gr00t

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "g1_control"))
from vla_env_gr00t import G1BrainCoPickEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(REPO_ROOT, "outputs", "g1_brainco_pick", "checkpoint-20000"),
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    # Matches the real training data's task string exactly (see
    # dataset_g1/brainco_pick/G1_Brainco_GraspRubiksCube_Dataset/meta/tasks.jsonl)
    # -- the model's predicted hand-closure deltas were measurably larger with
    # this exact phrasing vs. a generic paraphrase in a side-by-side check.
    parser.add_argument("--instruction", default="Pick up the Rubik's Cube and put it in the plate")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=15,
        help="Each chunk executes the model's full 16-step action horizon before re-querying. "
        "The real demonstrations are ~static for their first ~96 frames (6 chunks) and only "
        "start moving substantially around frame 100-200 (chunks 7-13), so short rollouts "
        "will look like nothing is happening even when the policy is working correctly.",
    )
    parser.add_argument(
        "--video-out",
        default=os.path.join(REPO_ROOT, "outputs", "g1_brainco_pick", "mujoco_pick_rollout.mp4"),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a live mujoco.viewer window instead of (also) saving a video. Needs a real "
        "display -- do not set MUJOCO_GL=egl when using this flag.",
    )
    args = parser.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    print(f"Loading Gr00tPolicy from {args.model_path} (embodiment={args.embodiment_tag}, device={args.device})...")
    policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=args.model_path, device=args.device)

    env = G1BrainCoPickEnv()
    obs = env.reset()

    frames = [obs["video"]["observation.images.head_stereo_left"][0, 0]]
    print(f"Rolling out {args.num_chunks} action chunk(s) -- instruction: '{args.instruction}'")
    print(f"Initial cube height: {env.cube_height():.3f} m")

    def run_chunks(viewer):
        nonlocal obs
        for chunk in range(args.num_chunks):
            obs["language"] = {"annotation.human.task_description": [[args.instruction]]}
            action, _ = policy.get_action(obs)
            arm_q_chunk = action["action.arm_q"][0]  # (16, 14)
            hand_cmd_chunk = action["action.hand_cmd"][0]  # (16, 12)

            for t in range(arm_q_chunk.shape[0]):
                obs = env.step(arm_q_chunk[t], hand_cmd_chunk[t], viewer=viewer)
                frames.append(obs["video"]["observation.images.head_stereo_left"][0, 0])

            print(f"  chunk {chunk + 1}/{args.num_chunks} done -- cube height: {env.cube_height():.3f} m")

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            run_chunks(viewer)
    else:
        run_chunks(viewer=None)

    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    import imageio

    imageio.mimsave(args.video_out, frames, fps=30, codec="libx264")
    print(f"Saved {len(frames)} frames to {args.video_out}")


if __name__ == "__main__":
    main()
