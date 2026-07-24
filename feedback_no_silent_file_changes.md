---
name: feedback-no-silent-file-changes
description: Never modify or overwrite a pre-existing file without flagging it first -- prefer new sibling files over in-place rewrites
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f713a406-c514-4aef-9edf-89b5cfad3029
  modified: 2026-07-23T10:59:52.709Z
---

Never silently modify or overwrite a file that already existed before this
session touched it. Before using Write/Edit on such a file, or before running
a command that mutates a shared/tracked file (`uv add`, config edits, etc.),
state clearly what's about to change and why, and wait for confirmation
rather than just doing it.

**Why:** In [[project-gr00t-mujoco-pick-integration]], `vla_env.py` (a
pre-existing file with a working single-arm/mock-inference implementation)
got overwritten via Write without being flagged first. The user had to stop
mid-task, ask "did I find it in the g1 folder?", and have it reverted via
`git restore`. Their explicit instruction afterward: "I don't want existing
files to be changed." A separate `uv add mujoco` change to `pyproject.toml`
was fine on its merits, but the user's objection was specifically that it
happened invisibly -- their follow-up: "This is OKAY, and I want it to be
updated. Just don't change anything silently in this session."

**How to apply:** When a task requires new capability that overlaps with an
existing file's purpose, default to creating a new sibling file (e.g.
`vla_env_gr00t.py` alongside the untouched original `vla_env.py`) and leave
the original alone, rather than rewriting it in place -- even if the new
version is strictly better. This applies to first touching any file with
prior content; it does NOT mean re-confirming every subsequent edit to a file
I created myself earlier in the same session (e.g. iterating on
`gr00t_pick_rubiks_cube.py` bug fixes didn't need re-permission each time --
only the first touch of something pre-existing does). Shared/tracked
dependency or config files (pyproject.toml, uv.lock) are fine to change when
the task needs it -- just say so before doing it, don't do it in the
background unannounced.
