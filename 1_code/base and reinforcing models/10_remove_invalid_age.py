"""
To prepare for the next steps in model training and testing, we need to 
identify records with invalid ages (NaN or empty age fields). Once identified, we can 
write the IDs of these records to a specified CSV file and remove them 
from the metadata file.
"""

import os
import sys
import pandas as pd
from hints_helpers import FileSystemUtils
from environment_variables import EnvVarsSingleton

NAME = '10_remove_invalid_age'
PROJECT = 'HINTS'

env_vars = EnvVarsSingleton.instance()
file_in  = sys.argv[1]
file_out = sys.argv[2]

print('**************')
print(file_in)
print(file_out)

fs_utils = FileSystemUtils()
fs_utils.make_dirs(NAME, PROJECT)

def find_invalid_records(csv_file, column_name, output_file):
    df = pd.read_csv(csv_file)
    invalid_records = df[df[column_name].isna() | df[column_name].eq("")]
    invalid_records.to_csv(output_file, index=False)

def remove_duplicate_ids(invalid_records, metadata, output_file):
    df1 = pd.read_csv(invalid_records)
    df2 = pd.read_csv(metadata)
    ids_to_remove = set(df1['isic_id'])
    df2 = df2[~df2['isic_id'].isin(ids_to_remove)]
    df2.to_csv(output_file, index=False)

find_invalid_records(env_vars.get_base_folder() + file_in, "age_approx", env_vars.get_pipeline_folder() + '/out/' + file_out)
filename = os.path.basename(file_in)
remove_duplicate_ids(env_vars.get_pipeline_folder() + '/out/' + file_out, env_vars.get_base_folder() + file_in, env_vars.get_pipeline_folder() + '/store/' + filename)