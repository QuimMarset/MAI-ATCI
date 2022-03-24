from utils.utils import exists_folder, create_folder, user_input_algorithm, user_input_mode
from constants.constants import PPO_NAME, RESULTS_PATH, WEIGHTS_PATH
from utils import run_PPO, run


if __name__ == "__main__":

    is_train_mode = user_input_mode()
    is_PPO = user_input_algorithm() 

    if not exists_folder(RESULTS_PATH):
        create_folder(RESULTS_PATH)
    if not exists_folder(WEIGHTS_PATH):
        create_folder(WEIGHTS_PATH)

    if is_PPO:
        train_function = run_PPO.train_agent
        test_function = run_PPO.test_agent
    else:
        train_function = run.train_agent
        test_function = run.test_agent

    if is_train_mode:
        train_function()
    else:
        test_function(render=True)