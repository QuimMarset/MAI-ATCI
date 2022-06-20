from utils.utils import *
from utils.run import train_agent, test_agent


if __name__ == "__main__":

    is_train_mode = user_input_mode()
    env_name = user_input_environment()

    if is_train_mode:
        results_path, weights_path = create_needed_folders(env_name)
        train_agent(results_path, weights_path, env_name)
    else:
        results_path, weights_path = get_last_execution_paths(env_name)
        test_agent(results_path, weights_path, env_name, render=False)