"""
Central configuration for the FT-CNN ECG arrhythmia classification pipeline.

Every numeric constant here is taken directly from the manuscript
(Abdullah et al., Biosensors 2026, 16, 326) so that the code is traceable
back to a specific section, table, or equation.
"""

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# §3.1 Dataset & preprocessing
# --------------------------------------------------------------------------- #
SAMPLING_RATE_HZ = 360
WINDOW_SAMPLES = 187          # 93 pre-R + 94 post-R = 187 (519 ms)
PRE_R_SAMPLES = 93
POST_R_SAMPLES = 94
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0
BANDPASS_ORDER = 4

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = len(AAMI_CLASSES)

# §3.1 augmentation bounds
AMPLITUDE_SCALE_RANGE = (0.9, 1.1)
TEMPORAL_WARP_RANGE = (0.8, 1.2)
MAX_RPEAK_SHIFT_SAMPLES = 5
MAX_QRS_AMPLITUDE_CHANGE = 0.20
TARGET_SAMPLES_PER_CLASS = 10000  # ~50,000 total training samples after augmentation

# §3.1 patient-wise (de Chazal et al.) split
DE_CHAZAL_TRAIN_RECORDS: List[int] = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
]
DE_CHAZAL_TEST_RECORDS: List[int] = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
    219, 221, 222, 228, 231, 232, 233, 234,
]
VAL_FRACTION_OF_TRAIN_PATIENTS = 0.10
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


# --------------------------------------------------------------------------- #
# §3.2 / Figure 4: FT-CNN architecture
# --------------------------------------------------------------------------- #
@dataclass
class FTCNNConfig:
    # Block 1: low-level morphology (P-wave / QRS)
    block1_filters: int = 32
    block1_kernel: int = 5
    # Block 2: mid-level inter-wave relationships
    block2_filters: int = 64
    block2_kernel: int = 3
    block2_spatial_dropout: float = 0.25
    # Block 3: high-level holistic synthesis
    block3_filters: int = 128
    block3_kernel: int = 3
    leaky_relu_alpha: float = 0.01
    # Fully connected head
    dense_1: int = 256
    dense_2: int = 128
    dense_2_dropout: float = 0.5
    dense_3: int = 64
    # Regularization
    l2_lambda: float = 1e-4
    # RR feature branch
    rr_feature_dim: int = 4
    num_classes: int = NUM_CLASSES
    input_length: int = WINDOW_SAMPLES
    softmax_temperature: float = 1.5


FT_CNN_CONFIG = FTCNNConfig()

# Kernel-size ablation grid (Table 10)
KERNEL_ABLATION_CONFIGS = {
    "fixed_3_3_3": (3, 3, 3),
    "fixed_5_5_5": (5, 5, 5),
    "proposed_5_3_3": (5, 3, 3),
    "larger_7_5_3": (7, 5, 3),
}


# --------------------------------------------------------------------------- #
# §3.3 Loss function (weighted categorical cross-entropy + focal loss)
# --------------------------------------------------------------------------- #
FOCAL_LOSS_GAMMA = 2.0
CLASS_WEIGHT_STRATEGY = "inverse_frequency"


# --------------------------------------------------------------------------- #
# §3.3 Training hyperparameters (FT-CNN)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    optimizer: str = "adam"
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_epsilon: float = 1e-7
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    cosine_annealing_cycle_epochs: int = 50
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 15  # FT-CNN
    benchmark_early_stopping_patience: int = 10  # baseline NN models
    benchmark_epochs: int = 50
    monitor: str = "val_loss"
    he_init_for_relu: bool = True
    glorot_init_output: bool = True


TRAIN_CONFIG = TrainConfig()


# --------------------------------------------------------------------------- #
# §4.4 Robustness analysis
# --------------------------------------------------------------------------- #
SNR_LEVELS_DB = [20, 10, 5]
ROBUSTNESS_AMPLITUDE_SCALE = 0.10       # +/-10%
ROBUSTNESS_TEMPORAL_SHIFT_SAMPLES = 10  # +/-10 samples

BASELINE_WANDER_FREQ_RANGE_HZ = (0.1, 1.0)
BASELINE_WANDER_AMPLITUDE_MAX_MV = 2.0
MOTION_ARTIFACT_FREQ_RANGE_HZ = (10, 100)
MOTION_ARTIFACT_AMPLITUDE_RANGE_MV = (0.5, 2.0)
MOTION_ARTIFACT_DURATION_MS = 50
POWERLINE_FREQ_HZ = 60  # or 50, region dependent
POWERLINE_AMPLITUDE_MV = 0.5


# --------------------------------------------------------------------------- #
# §4.5 XAI
# --------------------------------------------------------------------------- #
GRADCAM_TARGET_LAYER = "block3_conv"
IG_STEPS = 50
IG_BASELINE = "zero"
DELETION_TOP_FRACTION = 0.20
INSERTION_TOP_FRACTION = 0.30
SANITY_CHECK_NOISE_SNR_DB = 10

# Reported clinical alignment percentages from paper (used as reference targets
# in tests / sanity comparisons, NOT as ground truth to fit to)
REFERENCE_ATTRIBUTION_SHARE = {"QRS": 0.614, "P": 0.197, "T": 0.142, "baseline": 0.047}


RANDOM_SEED = 42
