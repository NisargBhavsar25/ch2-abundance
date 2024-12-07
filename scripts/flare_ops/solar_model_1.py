from xspec import *
import os
import time
from datetime import datetime
import os
from xspec import *
# Load local model
# modelfile = '../chspec'

# AllModels.lmod('chspec', dirPath=modelfile)

# Clear any previous data and models
AllData.clear()
AllModels.clear()
spec=None

def process_spectrum(date_str, l1dir=None, l2dir=None,fpath=".."):
    """
    Processes the spectrum for the given date.

    Parameters:
        date_str (str): Date in the format 'YYYYMMDD' (e.g., '20240826').
        l1dir (str): Optional base directory for Level 1 data. Defaults to generated path if not specified.
        l2dir (str): Optional base directory for Level 2 data. Defaults to generated path if not specified.
    """
    # Parse the date string
    start_time = time.perf_counter()

    date = datetime.strptime(date_str, "%Y%m%d")
    tref = datetime(2017, 1, 1)
    
    # Calculate start and stop times for spectrum generation
    tstart = (datetime(date.year, date.month, date.day, 1, 44) - tref).total_seconds()
    tstop = (datetime(date.year, date.month, date.day, 1, 47) - tref).total_seconds()
    
    # Generate default directories if not provided
    if l1dir is None:
        l1dir = f"{fpath}/xsm/data/{date.year}/{date.month:02d}/{date.day:02d}/raw"
    if l2dir is None:
        l2dir = f"{fpath}/xsm/data/{date.year}/{date.month:02d}/{date.day:02d}/calibrated"

    # Base filename for the data files
    base = f"ch2_xsm_{date.year}{date.month:02d}{date.day:02d}_v1"
    
    # File paths
    l1file = f"{l1dir}/{base}_level1.fits"
    hkfile = f"{l1dir}/{base}_level1.hk"
    safile = f"{l1dir}/{base}_level1.sa"
    gtifile = f"{l2dir}/{base}_level2.gti"
    
    # Output spectrum file
    specbase = f"ch2_xsm_{date.year}{date.month:02d}{date.day:02d}_l1"
    specfile = f"{specbase}.pha"
    
    # Generate spectrum command
    genspec_command = (
        f"xsmgenspec l1file={l1file} specfile={specfile} spectype='time-integrated' "
        f"tstart={tstart} tstop={tstop} hkfile={hkfile} safile={safile} gtifile={gtifile} tbinsize=1"
    )
    
    # Execute spectrum generation command
    os.system(genspec_command)
    end_time = time.perf_counter()
    print(f"Genspec took {end_time - start_time:.6f} seconds")
    print(f"Processing for date {date_str} completed.")

# Example usage:

def model_solar(specfile, backfile, modelfile, model_choice="chisoth"):
    """
    Sets up the XSPEC model for solar spectrum fitting.

    Parameters:
        specfile (str): Path to the .pha file.
        backfile (str): Path to the .rmf file.
        modelfile (str): Path to the model's directory.
        model_choice (str): Choice of model, either 'chisoth' or 'vvapec'.

    Returns:
        Model: The fitted model.
    """
    global spec
    start_time = time.perf_counter()

    
    # Load spectrum with background
    spec = Spectrum(specfile, backFile=backfile)

    # Set energy range for analysis
    AllModels.setEnergies("0.1,32,3000")
    scatter_model = Model("atable{scatter_atable.fits}")

    # Initialize and fit the chosen model
    if model_choice == "chisoth":
        m1 = Model("chisoth")
        Fit.perform()
        m1(12).frozen = False
        m1(13).frozen = False
        m1(14).frozen = False
        m1(16).frozen = False
    elif model_choice == "vvapec":
        m1 = Model("vvapec")
        Fit.perform()
        m1(13).frozen = False
        m1(14).frozen = False
        m1(15).frozen = False
        m1(17).frozen = False
    else:
        raise ValueError("Invalid model choice. Choose either 'chisoth' or 'vvapec'.")
    
    # Perform initial fit
    
    # Free abundances and re-fit
    Fit.perform()

    # Display fit parameters
    AllModels.show(parIDs="1 12 13 14 16 31")

    # Save model configuration to .xcm file
    # fxcm = specfile.replace('.pha', '_Fit.xcm')
    # if os.path.isfile(fxcm):
    #     os.remove(fxcm)
    # Xset.save(fxcm)
    # print("Model saved to", fxcm)
    end_time = time.perf_counter()
    print(f"Fitting took {end_time - start_time:.6f} seconds")
    return m1,spec


def calc_flux( output_filename, energy_start=0.1, energy_end=30.1, energy_step=0.01):
    global spec
    """
    Calculates the flux for the given model over specified energy ranges.

    Parameters:
        model (Model): The XSPEC model to calculate flux for.
        output_filename (str): File to write the flux results.
        energy_start (float): Starting energy in keV.
        energy_end (float): Ending energy in keV.
        energy_step (float): Energy step size in keV.
    """
    lst = []
    for e in range(0,3000,1):
        e=0.1+e/100
        AllModels.calcFlux(f"{e} {e+0.01}")
        lst.append([e,0,spec.flux[-3]/0.01])
        # Calculate flux for each small energy range
        # print()
      
    
    # Save calculated flux to file
    with open(output_filename, "w") as f:
        for item in lst:
            f.write(f"{item[0]:.6f}\t{item[1]:.6f}\t{item[2]:.6f}\n")
    
    print(f"Spectrum data saved to {output_filename}")

# Example usage:
# Set the paths for the input files and the model directory
dt="20210827"
# process_spectrum("20210827")
specbase = f"ch2_xsm_{dt}_l1"
specfile = specbase+'.pha'
backfile = '../class_rmf_v1.rmf'
specname=specbase.split("/")[-1]
# Choose the model
model_choice = "vvapec"  # Options: "chisoth" or "vvapec"
output_filename = "spectrum/"+specname+f"_{model_choice}.txt"
# Xset.logChatter = 0
# Xset.chatter = 0

# Set up the model
model_solar(specfile, backfile, modelfile, model_choice=model_choice)
# Calculate and save flux
calc_flux( output_filename, energy_start=0.1, energy_end=30.0, energy_step=0.01)
