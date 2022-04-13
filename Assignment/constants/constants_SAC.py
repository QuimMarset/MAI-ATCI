
# =======================
# Names constants
# =======================

ALGORITHM = "SAC"

# =======================
# Paths constants
# =======================

RESULTS_PATH = f'./results/{ALGORITHM}/'
WEIGHTS_PATH = f'./weights/{ALGORITHM}/'

# =======================
# Agent constants
# =======================

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 0.5
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
ITERATIONS = 10000
ITERATION_STEPS = BUFFER_SIZE

# =======================
# Agent constants
# =======================

BUFFER_SIZE = 1e6
# Entropy-regularized objective weight
ALPHA = 0.2
# Polyak average update
TAU = 0.005

# =======================
# Train constants
# =======================

BATCH_SIZE = 128
EPOCHS = 10000
STEPS_PER_EPOCH = 5000

SAMPLING_STEPS = 1000
UPDATE_EVERY = 50
START_UPDATING = 1000