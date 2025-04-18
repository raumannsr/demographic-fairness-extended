"""
Remove all duplicates (input) from the specified metadata file.
"""
import os
import sys
import pandas as pd
from hints_helpers import FileSystemUtils
from environment_variables import EnvVarsSingleton

NAME = '8_remove_duplicates'
PROJECT = 'HINTS'

env_vars = EnvVarsSingleton.instance()
fs_utils = FileSystemUtils()
fs_utils.make_dirs(NAME, PROJECT)

file_out  = sys.argv[1]

original_metadata_file = env_vars.get_meta_data_file()
duplicate_ids_file = os.path.join(env_vars.get_base_folder(), '2_pipeline/7_find_duplicates/out', env_vars.get_experiments_folder() + '.csv')
new_metadata_file = os.path.join(env_vars.get_pipeline_folder(), 'out', env_vars.get_experiments_folder() + '.csv')

original = pd.read_csv(original_metadata_file)
duplicates = pd.read_csv(duplicate_ids_file)
df_twins = pd.DataFrame(duplicates, columns=['hash', 'id1', 'id2'])
df_new = pd.DataFrame(original)

df_twins = df_twins.rename(columns={'id1': 'isic_id'})
merged_df = pd.merge(df_new, df_twins, on='isic_id', how='left', indicator=True)
result_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
result_df = result_df.iloc[:, :-2]
result_df.to_csv(os.path.join(env_vars.get_pipeline_folder(), 'store', file_out), index=False)