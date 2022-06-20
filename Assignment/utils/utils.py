import os
from constants import *


def user_input_mode():
    option = -1
    while option not in [1, 2]:
        option = int(input("1: Train\
                        \n2: Test saved agent\n"))

        if option not in [1, 2]:
            print("Invalid option\n")
    
    return option == 1
    

def user_input_environment():
    option = -1
    while option not in [1, 2, 3]:
        option = int(input("1: Lunar lander \
                        \n2: Bipedal walker \
                        \n3: VizDoom\n"))

        if option == 1:
            return LANDER
        elif option == 2:
            return BIPEDAL
        elif option == 3:
            return VIZDOOM
        else:
            print("Invalid option\n")


def exists_folder(path):
    return os.path.exists(path)


def create_folder(path):
    os.makedirs(path, exist_ok = True)


def is_folder_empty(path):
    return not any(os.scandir(path))


def create_needed_folders(env_name):
    results_path = os.path.join(RESULTS_PATH, env_name)
    weights_path = os.path.join(WEIGHTS_PATH, env_name)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    num_subfolders = len(os.listdir(results_path))

    results_path = os.path.join(results_path, f'execution_{num_subfolders+1}')
    weights_path = os.path.join(weights_path, f'execution_{num_subfolders+1}')

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    return results_path, weights_path


def get_last_execution_paths(env_name):
    results_path = os.path.join(RESULTS_PATH, env_name)
    weights_path = os.path.join(WEIGHTS_PATH, env_name)

    last_execution = os.listdir(weights_path)[-1]

    weights_path = os.path.join(weights_path, last_execution)
    results_path = os.path.join(results_path, last_execution)

    return results_path, weights_path