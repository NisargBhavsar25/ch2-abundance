import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def process_spectrum_conv(xmso_path, num_channels=2048, plot=True):
    channel_sums_conv = np.zeros(num_channels)
    channel_sums_unconv = np.zeros(num_channels)
    energies = np.zeros(num_channels)
    
    tree = ET.parse(xmso_path)
    root = tree.getroot()

    conv_root = root.find('spectrum_conv')
    
    for channel in conv_root.findall('channel'):
        channel_nr = int(channel.find('channelnr').text)
        counts = channel.findall('counts')
        energy = float(channel.find('energy').text)
        channel_sum = sum(float(count.text) for count in counts)
        channel_sums_conv[channel_nr] = channel_sum
        energies[channel_nr] = energy

    unconv_root = root.find('spectrum_unconv')

    for channel in unconv_root.findall('channel'):
        channel_nr = int(channel.find('channelnr').text)
        counts = channel.findall('counts')
        channel_sum = sum(float(count.text) for count in counts)
        channel_sums_unconv[channel_nr] = channel_sum
    
    if plot == True:
        plt.figure(figsize=(12, 6))  # Create a new figure for the first plot
        plt.subplot(1, 2, 1)  # First subplot
        plt.plot(energies, channel_sums_conv)
        plt.xlabel('Energy')
        plt.xticks(np.linspace(min(energies), max(energies), 20))
        plt.xticks(rotation=45)
        plt.ylabel('Sum of Counts')
        plt.title('spectrum_conv')

        plt.subplot(1, 2, 2)  # Second subplot
        plt.plot(energies, channel_sums_unconv, color='orange')
        plt.xlabel('Energy')
        plt.xticks(np.linspace(min(energies), max(energies), 20))
        plt.xticks(rotation=45)
        plt.ylabel('Counts')
        plt.title('spectrum_unconv')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig('channel_sums.png')  # Save the figure as a PNG file
        plt.close()  


    return channel_sums_conv, channel_sums_unconv


conv_sum, unconv_sum = process_spectrum_conv('test.xmso')
