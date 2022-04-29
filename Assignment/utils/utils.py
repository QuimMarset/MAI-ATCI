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
    

def exists_folder(path):
    return os.path.exists(path)


def create_folder(path):
    os.makedirs(path, exist_ok = True)


def is_folder_empty(path):
    return not any(os.scandir(path))


def create_needed_folders():
    results_path = os.path.join(RESULTS_PATH, ENVIRONMENT)
    weights_path = os.path.join(WEIGHTS_PATH, ENVIRONMENT)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    return results_path, weights_path