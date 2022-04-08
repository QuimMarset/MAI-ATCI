from utils.utils import create_folders_if_needed, user_input_algorithm, user_input_mode
from utils import run_PPO, run_SAC


if __name__ == "__main__":

    is_train_mode = user_input_mode()
    is_PPO = user_input_algorithm() 

    create_folders_if_needed()

    if is_PPO:
        train_function = run_PPO.train_agent
        test_function = run_PPO.test_agent
    else:
        train_function = run_SAC.train_agent
        test_function = run_SAC.test_agent

    if is_train_mode:
        train_function()
    else:
        test_function(render=True)