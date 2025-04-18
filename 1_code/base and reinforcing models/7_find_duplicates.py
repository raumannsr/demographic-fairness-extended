"""
Create a CSV file containing the IDs of the skin lesion duplicates located in a pre-specified folder.
"""
import os
import pandas as pd
import hashlib
from hints_helpers import FileSystemUtils
from environment_variables import EnvVarsSingleton

NAME = '7_find_duplicates'
PROJECT = 'HINTS'

def calculate_hash(file_path, algorithm='md5', block_size=65536):
    hash_function = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()

    with open(file_path, 'rb') as file:
        for block in iter(lambda: file.read(block_size), b''):
            hash_function.update(block)

    return hash_function.hexdigest()

def find_duplicate_files(directory, algorithm='md5'):
    hash_dict = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = calculate_hash(file_path, algorithm)

            if file_hash in hash_dict:
                hash_dict[file_hash].append(file_path)
            else:
                hash_dict[file_hash] = [file_path]

    return {key: value for key, value in hash_dict.items() if len(value) > 1}

env_vars = EnvVarsSingleton.instance()
fs_utils = FileSystemUtils()
fs_utils.make_dirs(NAME, PROJECT)

directory_path = env_vars.get_image_path()
duplicates = find_duplicate_files(directory_path, algorithm='md5')

if duplicates:
    df_ids = pd.DataFrame()
    print(f"No of duplicates found: {len(duplicates)}")
    for hash_value, files in duplicates.items():
        id1 = os.path.splitext(os.path.basename(files[0]))[0]
        id2 = os.path.splitext(os.path.basename(files[1]))[0]
        df_ids = df_ids.append({'id1': id1, 'id2': id2, 'hash': hash_value}, ignore_index=True)
    df_ids.to_csv(os.path.join(env_vars.get_pipeline_folder(), 'out', env_vars.get_experiments_folder() + '.csv'), index=False)
else:
    print("No duplicate files found.")