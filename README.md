# Unitree G1 MuJoCo Simulation & VLA Inference

MuJoCo simulation environment and control scripts for the Unitree G1 humanoid robot, including whole-body control, RL training, and VLA (vision-language-action) policy inference.

## Setup with uv

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management, with dependencies pinned in `uv.lock`.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify the install:

```bash
uv --version
```

### 2. Clone the repository

```bash
git clone <repo-url>
cd mechecm
```

### 3. Sync the environment

`uv sync` reads `pyproject.toml` and `uv.lock` and creates a `.venv` with the exact pinned dependency versions:

```bash
uv sync
```

This requires Python >= 3.13 (see `pyproject.toml`); `uv` will download a matching Python automatically if one isn't already installed.

### 4. Run scripts

Use `uv run` to execute scripts inside the managed environment without manually activating it:

```bash
uv run python scripts/run_simulation.py
uv run python scripts/vla_inference.py
uv run python scripts/visualize_g1.py
```

Alternatively, activate the virtual environment directly:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Adding or updating dependencies

```bash
uv add <package>       # add a new dependency and update uv.lock
uv lock --upgrade       # refresh the lockfile
```

## Project Structure

```text
├── config/            # Modality/config definitions
├── meshes/             # Robot STL meshes (Unitree G1 + BrainCo hands)
├── model_policy/       # Trained ONNX policies (stand, walk)
├── scenes/mujoco/      # MuJoCo scene XMLs
├── scripts/            # Entry point scripts (simulation, training, VLA inference)
├── src/g1_control/     # Core control / environment code
├── g1_fixed_floating.xml       # Full G1 MuJoCo model (floating base)
├── g1_fixed_manipulation.xml   # G1 MuJoCo model (fixed base, manipulation)
├── pyproject.toml
└── uv.lock
```
