from utils.utils import exists_folder, create_folder, user_input
from constants.constants import RESULTS_PATH, WEIGHTS_PATH
from utils.run import train_agent, test_agent


if __name__ == "__main__":

    is_train_mode = user_input()    

    if not exists_folder(RESULTS_PATH):
        create_folder(RESULTS_PATH)

    if not exists_folder(WEIGHTS_PATH):
        create_folder(WEIGHTS_PATH)

    if is_train_mode:
        train_agent()
    else:
        test_agent(render=True)