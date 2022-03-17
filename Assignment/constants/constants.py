# =======================
# Paths constants
# =======================

RESULTS_PATH = "./results/"
WEIGHTS_PATH = './weights/'

# =======================
# Names constants
# =======================

ENVIRONMENT = "BipedalWalker-v3"
ALGORITHM = "SAC"

# =======================
# Environment constants
# =======================

# Multi-environment training in PPO
NUM_ENVS = 6

# =======================
# Agent constants
# =======================

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 0.5
GAMMA = 0.99

# PPO

PPO_BUFFER_SIZE = 2000

# Generalized advantage estimator
GAE_LAMBDA = 0.95
# Clipped surrogate objective
EPSILON = 0.1
# Maximum KL-Divergence (used to stop gradient backpropagation)
MAX_KL_DIVERG = 0.03

# SAC

SAC_BUFFER_SIZE = 1e6
# Entropy-regularized objective weight
ALPHA = 0.2
# Polyak average update
TAU = 0.005

# =======================
# Train constants
# =======================

# PPO

PPO_EPOCHS = 10
ITERATIONS = 1000
ITERATION_STEPS = 256

# SAC

SAC_EPOCHS = 10000
STEPS_PER_EPOCH = 5000

SAMPLING_STEPS = 10000

TRAIN_EXPERIMENTS = 1
BATCH_SIZE = 32

# =======================
# Test constants
# =======================

TEST_EPISODES = 100