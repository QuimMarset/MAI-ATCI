
# Paths constants

RESULTS_PATH = "./results/"
WEIGHTS_PATH = './weights/'


# Names constants

ENVIRONMENT = "BipedalWalker-v3"
ALGORITHM = "SAC"


# Environment constants


# Agent constants

LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 0.5
BUFFER_SIZE = 1e6
GAMMA = 0.99
# Entropy-regularized objective weight
ALPHA = 0.2
# Polyak average update
TAU = 0.005


# Train constants

EPOCHS = 10000
STEPS_PER_EPOCH = 5000

SAMPLING_STEPS = 10000

TRAIN_EXPERIMENTS = 1
BATCH_SIZE = 32


# Test constants

TEST_EPISODES = 100