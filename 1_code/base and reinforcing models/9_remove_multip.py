"""
Remove all multiplets (input) from the specified metadata file.
"""
import sys
import os
import pandas as pd
from hints_helpers import FileSystemUtils
from environment_variables import EnvVarsSingleton

NAME = '9_remove_multip'
PROJECT = 'HINTS'

env_vars = EnvVarsSingleton.instance()
duplicates_out = sys.argv[1]

fs_utils = FileSystemUtils()
fs_utils.make_dirs(NAME, PROJECT)

def remove_duplicate_patient_ids(metadata_in, duplicates_out, metadata_out):
    df = pd.read_csv(metadata_in)
    df = df[df['patient_id'].isnull() | ~df.duplicated(subset='patient_id', keep='first')]
    df.to_csv(metadata_out, index=False)

    original_df = pd.read_csv(metadata_in)
    result_df = pd.read_csv(metadata_out)
    difference_df = pd.concat([original_df, result_df]).drop_duplicates(keep=False)
    difference_df.to_csv(duplicates_out, index=False)

metadata_out = env_vars.get_pipeline_folder() + '/store/' +  os.path.basename(env_vars.get_meta_data_file())
duplicates_out = env_vars.get_pipeline_folder() + '/out/' +  duplicates_out
remove_duplicate_patient_ids(env_vars.get_meta_data_file(), duplicates_out, metadata_out)