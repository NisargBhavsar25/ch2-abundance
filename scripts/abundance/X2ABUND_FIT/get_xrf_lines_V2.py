from common_modules import *
import xraylib
import os
import numpy as np

def get_xrf_lines(at_no, k_shell, k_lines, l1_shell, l1_lines, l2_shell, l2_lines, l3_shell, l3_lines) -> Xrf_Lines:
    # Initialize the number of elements and empty arrays for results
    no_elements = len(at_no)
    edgeenergy = np.zeros((no_elements, 5))
    fluoryield = np.zeros((no_elements, 5))
    jumpfactor = np.zeros((no_elements, 5))
    radrate = np.zeros((no_elements, 5))
    lineenergy = np.zeros((no_elements, 5))
    
    # Constants for NIST cross-sections and related data
    energy_nist = np.zeros((no_elements, 100))
    photoncs_nist = np.zeros((no_elements, 100))
    totalcs_nist = np.zeros((no_elements, 100))
    elename_string = np.empty(no_elements, dtype=object)
    
    # Read the constants file once
    file_data = readcol('./data_constants/kalpha_be_density_kbeta.txt', format='I,F,A,F,F,F')
    atomic_number_list, kalpha_list, ele_list, be_list, density_list, kbeta_list = file_data
    
    # Base path for attenuation coefficient files
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Process each element
    for i, atomic_num in enumerate(at_no):
        # Identify element name and corresponding NIST data file
        element_index = np.where(atomic_number_list == atomic_num)[0][0]
        elename_string[i] = ele_list[element_index]
        
        # Load NIST cross-section data for each element from FFAST database
        filename = f"{script_path}/data_constants/ffast/ffast_{int(atomic_num)}_{elename_string[i]}.txt"
        try:
            column_data = readcol(filename, format='D,F,F,F,F,F,F,F')
            energy_nist[i, :len(column_data[0])] = column_data[0]
            photoncs_nist[i, :len(column_data[3])] = column_data[3]
            totalcs_nist[i, :len(column_data[5])] = column_data[5]
        except FileNotFoundError:
            continue
        
        # Edge energies for K and L shells
        edgeenergy[i, 0:2] = xraylib.EdgeEnergy(atomic_num, k_shell)
        edgeenergy[i, 2] = xraylib.EdgeEnergy(atomic_num, l1_shell)
        edgeenergy[i, 3] = xraylib.EdgeEnergy(atomic_num, l2_shell)
        edgeenergy[i, 4] = xraylib.EdgeEnergy(atomic_num, l3_shell)

        # Fluorescent yields with error handling for missing values
        shells = [k_shell, l1_shell, l2_shell, l3_shell]
        for idx, shell in enumerate([0, 2, 3, 4]):
            try:
                fluoryield[i, shell] = xraylib.FluorYield(atomic_num, shells[idx])
            except:
                fluoryield[i, shell] = 0.0

        # Jump factors for each shell
        for idx, shell in enumerate([0, 2, 3, 4]):
            try:
                jumpfactor[i, shell] = xraylib.JumpFactor(atomic_num, shells[idx])
            except:
                jumpfactor[i, shell] = 0.0

        # Function to calculate weighted averages of radiative rates and line energies
        def compute_weighted_average(lines):
            radiative_rates = np.array([xraylib.RadRate(atomic_num, line) for line in lines])
            line_energies = np.array([xraylib.LineEnergy(atomic_num, line) for line in lines])
            mask = radiative_rates > 0
            if mask.any():
                weighted_energy = np.sum(radiative_rates[mask] * line_energies[mask]) / np.sum(radiative_rates[mask])
                total_radrate = np.sum(radiative_rates[mask])
                return weighted_energy, total_radrate
            return 0.0, 0.0

        # Calculate for K-beta, K-alpha, L1, L2, L3 lines and store results
        lineenergy[i, 0], radrate[i, 0] = compute_weighted_average(k_lines[3:8])  # K-beta
        lineenergy[i, 1], radrate[i, 1] = compute_weighted_average(k_lines[0:3])  # K-alpha
        lineenergy[i, 2], radrate[i, 2] = compute_weighted_average(l1_lines)      # L1
        lineenergy[i, 3], radrate[i, 3] = compute_weighted_average(l2_lines)      # L2
        lineenergy[i, 4], radrate[i, 4] = compute_weighted_average(l3_lines)      # L3

    # Return structured data containing all calculated constants
    return Xrf_Lines(edgeenergy, fluoryield, jumpfactor, radrate, lineenergy, energy_nist, photoncs_nist, totalcs_nist, elename_string)
