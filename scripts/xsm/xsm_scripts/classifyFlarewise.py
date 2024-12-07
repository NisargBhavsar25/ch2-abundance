import pandas as pd
import numpy as np

"""
This script is used to classify the flares based on the flare_class_with_bg column.
"""

df = pd.read_csv('BallTree_sorted.csv')

# Remove rows where flare_class is NaN
df_cleaned = df.dropna(subset=['flare_class_with_bg'])

# Handle flare_class - Extract letter and number, including 'sub-' cases
def extract_flare_class_details(flare_class):
    # Check if the flare_class has 'sub-' prefix
    if 'sub-' in flare_class:
        # Remove the 'sub-' part and treat the rest as the letter
        letter_part = flare_class.replace('sub-', '')
        number_part = 0  # No number for 'sub-'
    else:
        # Normal case like A1, A2, X, etc.
        letter_part = flare_class[:1]  # Extract the letter part (first character)
        number_part = flare_class[1:] if flare_class[1:].isdigit() else 0  # Extract the number part if exists
        number_part = int(number_part) if isinstance(number_part, str) and number_part.isdigit() else 0

    return letter_part, number_part

# Apply the function to extract flare_class letter and number
df_cleaned[['flare_class_letter', 'flare_class_number']] = df_cleaned['flare_class_with_bg'].apply(
    lambda x: pd.Series(extract_flare_class_details(x))
)

# Sort the DataFrame first by group_id, then by flare_class_letter and flare_class_number
df_sorted = df_cleaned.sort_values(by=['group_id', 'flare_class_letter', 'flare_class_number'])

# Create a new subgroup_id based on both group_id and flare_class
# Group by both group_id and flare_class (letter and number), then assign a unique ID within each group
df_sorted['subgroup_id'] = df_sorted.groupby(['group_id', 'flare_class_letter', 'flare_class_number']).ngroup()
df_sorted = df_sorted.drop(columns=['flare_class_number','flare_class_letter'])

"""
  Use only one of the below, comment the other
"""
# Remove duplicate STARTIME within each subgroup_id
df_unique_start_time = df_sorted.drop_duplicates(subset=['subgroup_id', 'STARTIME'], keep='first')
# Remove duplicate post_fit_start_time within each subgroup_id
df_unique_start_time = df_sorted.drop_duplicates(subset=['subgroup_id', 'post_fit_start_time'], keep='first')

"""# Save the sorted DataFrame with the new subgroup_id to a new CSV
output_filename = 'sorted_with_subgroup_id_and_subflare_class.csv'
df_sorted.to_csv(output_filename, index=False)"""

# Print the output (optional)
#print(f"Sorted locations with new subgroup IDs saved to {output_filename}")
#print(df_unique_start_time.head())  # Display the first few rows to verify
