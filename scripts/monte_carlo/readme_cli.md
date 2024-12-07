generate_input.py : code written by yash sirvi on generating xmsi files   
    
generator.py : trial code written to test how to make files from csv rows abundance values   
   
sampled100.csv: Sampled 100 rows from the abundances csv which is uploaded to drive at the same location as updated_final_output.csv

ele_abund_lpgrs.py : This code is modified from the original code which was unoptimised. now it takes less than 20s to run over the whole csv.

subprocessor.py : trial code written to understand CLI simulations and directory flow.

subprocess_cli.py: the final code file, that runs as
```bash
python subprocess_cli.py <csv_file> <txt_file>
```
generates both input and output files in 2 different directories, checks if the element db is present in the curernt directory or not otherwise generates one.   