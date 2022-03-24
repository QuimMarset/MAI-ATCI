import os
from constants.constants import PPO_NAME, SAC_NAME


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
        option = int(input(f'1: {PPO_NAME}\n'
                    + f'2: {SAC_NAME}\n'))

        if option not in [1, 2]:
            print("Invalid option\n")
    
    return option == 1


def exists_folder(path):
    return os.path.exists(path)


def create_folder(path):
    os.makedirs(path, exist_ok = True)


def is_folder_empty(path):
    return not any(os.scandir(path))