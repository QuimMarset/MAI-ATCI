import os
from constants import constants_PPO, constants_SAC
from constants.constants_general import ENVIRONMENT


def user_input_mode():
    option = -1
    while option not in [1, 2]:
        option = int(input("1: Train\
                        \n2: Test saved agent\n"))

        if option not in [1, 2]:
            print("Invalid option\n")
    
    return option == 1
    

def user_input_algorithm():
    option = -1
    while option not in [1, 2]:
        option = int(input(f'1: {constants_PPO.ALGORITHM}\n'
                    + f'2: {constants_SAC.ALGORITHM}\n'))

        if option not in [1, 2]:
            print("Invalid option\n")
    
    return option == 1


def exists_folder(path):
    return os.path.exists(path)


def create_folder(path):
    os.makedirs(path, exist_ok = True)


def is_folder_empty(path):
    return not any(os.scandir(path))


def create_folders_if_needed():
    results_path = os.path.join(constants_PPO.RESULTS_PATH, ENVIRONMENT)
    weights_path = os.path.join(constants_PPO.WEIGHTS_PATH, ENVIRONMENT)

    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)

    return results_path, weights_path