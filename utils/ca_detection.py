import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats as scs
import os
import json
import sys
from glob import glob

    # Directory containing
def get_spectrum_data(pha_file):
    """
    Extract channel and counts data from a PHA spectrum file.
    
    Args:
        pha_file (str): Path to the PHA FITS file
        
    Returns:
        tuple: (channels, counts) arrays containing the spectrum data
    """
    # Open the PHA file
    with fits.open(pha_file) as hdul:
        # Locate the SPECTRUM extension
        spectrum_hdu = None
        for hdu in hdul:
            if hdu.header.get("EXTNAME") == "SPECTRUM":
                spectrum_hdu = hdu
                break
        
        # Extract CHANNEL and COUNTS data
        data = spectrum_hdu.data
        channels = data["CHANNEL"] * 0.0135
        counts = data["COUNTS"]
    
    return channels, counts

def plot_spectrum(channels, counts):
    """Plot the PHA spectrum"""
    plt.figure(figsize=(10, 6))
    plt.step(channels, counts, where="mid", color="blue", label="PHA Spectrum")
    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.xlim(0, 700 * 0.0135)
    plt.title("PHA Spectrum Visualization")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

def window_spectrum(counts, channels, min_channel=2.5, max_channel=5.5):
    """
    Plot counts vs channels within a specified channel range
    
    Parameters:
    -----------
    counts : array-like
        Count values
    channels : array-like
        Channel values
    min_channel : float, optional
        Minimum channel value to plot (default: 2.5)
    max_channel : float, optional 
        Maximum channel value to plot (default: 5.5)
        
    Returns:
    --------
    array-like: Filtered channels array
    """
    # Apply the channel scaling factor
    
    
    mask = (channels >= min_channel) & (channels <= max_channel)
    
    # Apply the mask to filter channels
    filtered_counts = counts[mask]
    
    # Check if the filtered channels are empty
    if len(filtered_counts) == 0:
        print("Warning: No data in the specified channel range.")
        return np.array([])
    
    return filtered_counts

def detect_peaks(counts, n=3):
    """
    Detect peaks in the counts data using n-sigma method.
    
    Parameters:
    -----------
    counts : array-like
        Count values
    n : int, optional
        Number of standard deviations for peak detection (default: 3)
        
    Returns:
    --------
    tuple: (peak_regions, peaks) where peak_regions is a list of [start, end] indices and peaks is a list of peak indices
    """
    peak_regions = []
    peaks = []
    mean, _, sigma = scs(counts[~np.isnan(counts)])
    t_start_idx = 0
    t_end_idx = t_start_idx
    j = 0
    temp = 0
    for i in range(len(counts)):
        if counts[i] < mean + n * sigma:
            if j != 0:
                t_end_idx = temp
                if j > 4:
                    peak_regions.append([t_start_idx, t_end_idx])
                    peaks.append((t_start_idx + t_end_idx) // 2)
            j = 0
        elif j == 0:
            if not np.isnan(counts[i]):
                t_start_idx = i
                temp = t_start_idx
                j += 1
        else:
            if not np.isnan(counts[i]):
                temp = i
                j += 1
            if i == len(counts) - 1:
                t_end_idx = i
                if j > 4:
                    peak_regions.append([t_start_idx, t_end_idx])
                    peaks.append((t_start_idx + t_end_idx) // 2)

    return peak_regions, peaks

def detect_calcium(pha_file, output_dir="/home/subarno/Desktop/isro/elemental-super-resolution/ch2-abundance/fits_utils/CLASS/output_archive"):

    results = {}
    channels, counts = get_spectrum_data(pha_file)
    filtered_counts = window_spectrum(counts, channels)
    
    peak_regions, peaks = detect_peaks(filtered_counts)
    detection = len(peaks) > 0

    results[pha_file] = {
        'peaks': peaks,
        'peak_regions': peak_regions
    }

    # Save results to a JSON file
    output_file = os.path.join(output_dir, f"{os.path.basename(pha_file)}_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return detection, results


if __name__ == "__main__":

    pha_directory = "/home/subarno/Desktop/isro/elemental-super-resolution/ch2-abundance/fits_utils/CLASS/output_archive"
    pha_files = glob(os.path.join(pha_directory, "*.fits"))
    detection, results = detect_calcium(pha_files[0])

    print(detection)
     
    # 'peaks': peaks.tolist(),
    #     'peak_regions': peak_regions
    # }
