# =======================
# Paths constants
# =======================

RESULTS_PATH = "./results/"
WEIGHTS_PATH = './weights/'

# =======================
# Names constants
# =======================

ENVIRONMENT = "MountainCarContinuous-v0"
SAC_NAME = "SAC"
PPO_NAME = "PPO"

# =======================
# Environment constants
# =======================

# Multi-environment training in PPO
NUM_ENVS = 5

# =======================
# Agent constants
# =======================

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 0.5
GAMMA = 0.99

# PPO

PPO_BUFFER_SIZE = 512

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

TRAIN_EXPERIMENTS = 1
BATCH_SIZE = 256

# PPO

PPO_EPOCHS = 10
ITERATIONS = 10000
ITERATION_STEPS = PPO_BUFFER_SIZE

# SAC

SAC_EPOCHS = 10000
STEPS_PER_EPOCH = 5000

SAMPLING_STEPS = 10000
UPDATE_EVERY = 50
START_UPDATING = 1000

# =======================
# Test constants
# =======================

TEST_EPISODES = 100