from utils.utils import create_needed_folders, user_input_mode
from utils.run import train_agent, test_agent


if __name__ == "__main__":

    is_train_mode = user_input_mode()

    results_path, weights_path = create_needed_folders()

    if is_train_mode:
        train_agent(results_path, weights_path)
    else:
        test_agent(results_path, weights_path, render=True)