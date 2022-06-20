
# =======================
# Names constants
# =======================

BIPEDAL = 'BipedalWalker-v3'
LANDER = 'LunarLanderContinuous-v2'
# Change to the desired config file inside environments/vizdoom_files
VIZDOOM = 'basic'

ALGORITHM = 'PPO'

# =======================
# Paths constants
# =======================

RESULTS_PATH = './results/'
WEIGHTS_PATH = './weights/'

VIZDOOM_CONFIGS_PATH = './environments/vizdoom_files/'

# =======================
# Environment constants
# =======================

REWARD_SCALE = 0.01

FRAMES_STACKED = 4
FRAMES_SKIPPED = 4
FRAME_RESIZE = (100, 100)

# Multi-environment training
NUM_ENVS = 8

# =======================
# Agent constants
# =======================

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 0.5
GAMMA = 0.99

BUFFER_SIZE = 128
# Generalized advantage estimator
GAE_LAMBDA = 0.95
# Clipped surrogate objective
EPSILON = 0.2
# Maximum KL-Divergence (used to stop gradient backpropagation)
MAX_KL_DIVERG = 0.25
# Batch epochs
EPOCHS = 5

# =======================
# Train constants
# =======================

TRAIN_EXPERIMENTS = 3
TRAIN_EPISODES = 3000

BATCH_SIZE = 64
ITERATIONS = 1000
ITERATION_STEPS = BUFFER_SIZE

# =======================
# Test constants
# =======================

TEST_EPISODES = 100