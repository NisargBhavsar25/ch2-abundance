import os
import matplotlib.pyplot as plt
from astropy.io import fits

"""
This script visualizes the PHA files in the BKGs directory and saves the plots in the mergedPHAs directory.
"""

# Directory containing the PHA files
pha_directory = "/root/code/BKGs"
output_directory = "mergedPHAs"
count = 0

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the directory
for file_name in os.listdir(pha_directory):
    if file_name.endswith(".pha"):
        pha_file = os.path.join(pha_directory, file_name)
        print(f"Processing {pha_file}...")

        try:
            # Open the PHA file
            with fits.open(pha_file) as hdul:
                # Locate the SPECTRUM extension
                spectrum_hdu = None
                for hdu in hdul:
                    if hdu.header.get("EXTNAME") == "SPECTRUM":
                        spectrum_hdu = hdu
                        break

                if spectrum_hdu is None:
                    print(f"Warning: SPECTRUM extension not found in {file_name}. Skipping...")
                    continue

                # Extract CHANNEL and COUNTS data
                data = spectrum_hdu.data
                channels = data["CHANNEL"]
                counts = data["COUNTS"]

            # Plot the spectrum
            plt.figure(figsize=(10, 6))
            plt.step(channels, counts, where="mid", color="blue", label=f"PHA Spectrum: {file_name}")
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.title(f"PHA Spectrum Visualization for {file_name}")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            print(f"Plotting {file_name}...")

            # Save the plot to the output directory
            plot_path = os.path.join(output_directory, f"{file_name}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            count += 1
            plt.close()
            if count > 10:
                break

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
