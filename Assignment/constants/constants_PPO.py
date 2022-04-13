from constants.constants_general import BASE_RESULTS_PATH, BASE_WEIGHTS_PATH


# =======================
# Names constants
# =======================

ALGORITHM = "PPO"

# =======================
# Paths constants
# =======================

RESULTS_PATH = f'{BASE_RESULTS_PATH}/{ALGORITHM}/'
WEIGHTS_PATH = f'{BASE_WEIGHTS_PATH}/{ALGORITHM}/'

# =======================
# Environment constants
# =======================

# Multi-environment training
NUM_ENVS = 6

# =======================
# Agent constants
# =======================

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 0.8
GAMMA = 0.99

BUFFER_SIZE = 512
# Generalized advantage estimator
GAE_LAMBDA = 0.95
# Clipped surrogate objective
EPSILON = 0.2
# Maximum KL-Divergence (used to stop gradient backpropagation)
MAX_KL_DIVERG = 0.03

# =======================
# Train constants
# =======================

BATCH_SIZE = 512
EPOCHS = 10
ITERATIONS = 2000
ITERATION_STEPS = BUFFER_SIZE