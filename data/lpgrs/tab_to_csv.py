import csv

# Specify the input and output file paths
input_file = 'lpgrs_high1_elem_abundance_20deg.tab' #change this
output_file = 'output_file.csv'

with open(input_file, 'r') as infile:
    lines = infile.readlines()

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Iterate over each line in the .tab file
    for line in lines:
        # Split the line into a list by spaces (ignoring leading/trailing spaces)
        row = line.strip().split()
        
        writer.writerow(row)

print("Conversion complete!")
