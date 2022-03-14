import os


def user_input():
    option = -1
    while option not in [1, 2]:
        option = int(input("1: Train\
                        \n2: Test saved agent\n"))

        if option not in [1, 2, 3]:
            print("Invalid option\n")
    
    return option
    

def exists_folder(path):
    return os.path.exists(path)


def create_folder(path):
    os.makedirs(path, exist_ok = True)


def is_folder_empty(path):
    return not any(os.scandir(path))