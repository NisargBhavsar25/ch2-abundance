import subprocess

def addPHAs(infile, outfile):
    cmd = f'addspec {infile} {outfile} false false & exit'
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"PHAs added successfully. Output saved to {outfile}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while adding PHAs: {e}")

# Example usage
"""
infile = '/root/code/grp/list2.txt' #Add the path to the input file
outfile = 'combined' #Add the root name of the output file

addPHAs(infile, outfile)
"""
