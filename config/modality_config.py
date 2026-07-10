# modality_config.py
# Modality configuration mapping for GR00T-N1.7-G1-BrainCo-Pick model inference

class ModalityConfig:
    def __init__(self, delta_indices, modality_keys, action_configs=None):
        self.delta_indices = delta_indices
        self.modality_keys = modality_keys
        self.action_configs = action_configs

class ActionConfig:
    def __init__(self, rep, type, state_key=None):
        self.rep = rep
        self.type = type
        self.state_key = state_key

# We match the exact configurations loaded by JeffrinSam/GR00T-N1.7-G1-BrainCo-Pick
config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.images.head_stereo_left"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "observation.state.arm_q",       # 14D arm joint positions (rad)
            "observation.state.hand_state",  # 12D BrainCo finger state
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),       # 16-step prediction horizon
        modality_keys=[
            "action.arm_q",    # 14D relative arm joint targets
            "action.hand_cmd", # 12D absolute hand commands
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}
