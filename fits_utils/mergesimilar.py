import pandas as pd
import os
from conv import convert_to_pha
from mergeconverted import merge_phas

df = pd.read_csv('grouped.csv')
no_grps = len(df['group_id'].unique())
subgroup_count = 0
parent_directory = '/root/code/grp/mergedPHAs'
flare_types = ['sub-A','A1','A2','A3','A4','A5','A6','A7','A8','A9','B1','B2','B3','B4','B5','B6','B7','B8','B9','C1','C2','C3','C4','C5','C6','C7','C8','C9','M1','M2','M3','M4','M5','M6','M7','X']

for i in range(0, no_grps):
    grp_df = df[df['group_id'] == i]
    group_dir_name = 'fits_' + grp_df["Centroid_LAT"][0].to_string() + '_' + grp_df["Centroid_LON"][0].to_string()
    new_dir_path = os.path.join(parent_directory, group_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)
    for flare_type in flare_types:
        if grp_df[grp_df['flare_type'] == flare_type].empty:
            continue
        else:
            # Create an empty text file for each flare_type
            flare_file_name = f'{flare_type}.txt'
            flare_file_path = os.path.join(new_dir_path, flare_file_name)
            
            # Just create the file without writing anything to it
            with open(flare_file_path, 'w') as file:
                pass
            
    for j in range(0, len(grp_df)):
        convert_to_pha('add_path_to_fits', parent_directory + '/' + group_dir_name + '/' + 'ch2_cla_l1' + 'add_complete_path' , parent_directory + '/' + group_dir_name + '/' + grp_df.loc[j, 'flare_type_without_bg'] + '.txt')
    
    os.makedirs(parent_directory + '/' + group_dir_name + '/' + 'flare_wise', exist_ok=True)
        
    for filename in os.listdir(parent_directory + '/' + group_dir_name):
        if filename.endswith('.txt'):
            file_path = os.path.join(parent_directory + '/' + group_dir_name, filename)
            merge_phas(file_path, group_dir_name, parent_directory + '/' + group_dir_name + '/' + 'flare_wise')