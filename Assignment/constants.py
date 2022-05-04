
# =======================
# Names constants
# =======================

ENVIRONMENT = "BipedalWalker-v3"
# LunarLanderContinuous-v2
ALGORITHM = "PPO"

# =======================
# Paths constants
# =======================

RESULTS_PATH = f'./results/'
WEIGHTS_PATH = f'./weights/'

# =======================
# Environment constants
# =======================

# Multi-environment training
NUM_ENVS = 8

# =======================
# Agent constants
# =======================

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 50.0
GAMMA = 0.99

BUFFER_SIZE = 512
# Generalized advantage estimator
GAE_LAMBDA = 0.95
# Clipped surrogate objective
EPSILON = 0.2
# Maximum KL-Divergence (used to stop gradient backpropagation)
MAX_KL_DIVERG = 0.15
# Batch epochs
EPOCHS = 10

# =======================
# Train constants
# =======================

TRAIN_EXPERIMENTS = 3
TRAIN_EPISODES = 5000

BATCH_SIZE = 512
ITERATIONS = 2000
ITERATION_STEPS = BUFFER_SIZE

# =======================
# Test constants
# =======================

TEST_EPISODES = 100