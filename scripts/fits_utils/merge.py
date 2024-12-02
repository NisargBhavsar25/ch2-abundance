import os
import pandas as pd
import numpy as np
from conv import convert_to_pha
from addPHAs import add_phas

"""
This script merges the PHAs of a given class file into a single PHA file.
"""

def create_directory_and_file(dir_name, file_name, string_list):
    """
    Creates a directory and writes strings to a text file within it.
    
    Args:
        dir_name (str): Name of the directory to create
        file_name (str): Name of the text file
        strings (list): List of strings to write
    """
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(dir_name, file_name)
    
    # Write strings to file
    with open(file_path, 'w') as file:
        for string in string_list:
            file.write('ch2_cla_l1_' + string + '.pha\n')

def merge_grps(input_csv, output_dir):
    columns = ['class_file_name','preceding_noise_files']
    df = pd.read_csv(input_csv, usecols=columns)
    index = 0
    
    for noise_file in df.iloc[:,1]:
        if (type(noise_file) != str):
            index += 1
            continue
        noise_list = noise_file.split(',')
        
        create_directory_and_file(df.iloc[index,0] + '_noise', df.iloc[index,0] + '.txt', noise_list)
        for noise in noise_list:
            noise_year = noise[0:4]
            noise_month = noise[4:6]
            noise_day = noise[6:8]
            convert_to_pha('/home/ubuntu/data/class/fits/cla/data/calibrated/'+noise_year+'/'+noise_month+'/'+noise_day+'/ch2_cla_l1_'+noise+'.fits', df.iloc[index,0] + '_noise/ch2_cla_l1_' + noise + '.pha')
        
        add_phas(df.iloc[index,0] + '_noise/ch2_cla_l1_' + df.iloc[index,0] + '.txt', output_dir + '/ch2_cla_l1_' + df.iloc[index,0] + '_BKJ.pha')
        index += 1
        
input_csv = 'class_files.csv' # Replace with your input CSV filename
output_dir = 'merged' # Replace with the path to the output directory
merge_grps(input_csv, output_dir) 
