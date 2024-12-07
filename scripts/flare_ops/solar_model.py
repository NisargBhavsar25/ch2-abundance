from xspec import *
import os
import time
from datetime import datetime
import os
from xspec import *
# Load local model
from astropy.io import fits
import os
import shutil
#ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits
#modelop_20210827T210316000_20210827T210332000.txt


# Specify the destination folder (make sure it exists)
destination_folder = 'xsm_files'

# Move the file


modelfile = '../chspec'

# AllModels.lmod('chspec', dirPath=modelfile)


spec=None
import subprocess
def process_spectrum(date_str, l1dir=None, l2dir=None,fpath="/home/heasoft/data/xsm/xsm_all/xsm/xsm/data"):
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
    tstart = (datetime(date.year, date.month, date.day, 0, 1,0) - tref).total_seconds()
    tstop = (datetime(date.year, date.month, date.day, 23, 59,59) - tref).total_seconds()
    # Generate default directories if not provided
    if l1dir is None:
        l1dir = f"{fpath}/{date.year}/{date.month:02d}/{date.day:02d}/raw"
    if l2dir is None:
        l2dir = f"{fpath}/{date.year}/{date.month:02d}/{date.day:02d}/calibrated"

    # Base filename for the data files
    base = f"ch2_xsm_{date.year}{date.month:02d}{date.day:02d}_v1"
    print(base)
    
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

    try:
    # Start the process
        process = subprocess.Popen(genspec_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Read output and error streams
        stdout, stderr = process.communicate(timeout=100)  # Set a timeout as needed

        # Check for errors in stderr or non-zero exit code
        if process.returncode != 0:
            print("An error occurred:")
            print(stderr,stdout)
            return False
        else:
            print("Command executed successfully:")
            print(stdout)
            end_time = time.perf_counter()
            print(f"Genspec took {end_time - start_time:.6f} seconds")
            print(f"Processing for date {date_str} completed.")

            return True
            
    except subprocess.TimeoutExpired:
        # If the command stops or waits for user input
        process.kill()
        print("Process killed due to timeout or unexpected user prompt.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    # genspec_command = (
    #     f"xsmgenspec l1file={l1file} specfile={specfile} spectype='time-integrated' "
    #     f"tstart={tstart} tstop={tstop} hkfile={hkfile} safile={safile} gtifile={gtifile} tbinsize=1"
    # )
    
    # Execute spectrum generation command
    # os.system(genspec_command)

# Example usage:

def model_solar(specfile, backfile, tablefile, model_choice="chisoth"):
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
    # Clear any previous data and models
    AllData.clear()
    AllModels.clear()
    
    # Load spectrum with background
    spec = Spectrum(specfile, backFile=backfile)

    # Set energy range for analysis

    m1 = Model(model_choice+"+constant*atable{tbmodel.fits}")
    # print("here after model")
    AllModels.setEnergies("0.1,32,3000")
    # print("here after set energy")
    # m1 = Model(model_choice)
    # sf = Model("constant")
    # m1(34).frozen=False
    # m1 = Model(model_choice)
    # sf.constant.factor = 1.0  # Starting value
    # sf.constant.factor.frozen = False

    # s1.atable.par1 = 1.0  # Example: scaling factor for table model
    # s1.atable.par1.frozen = False  # Free the parameter to vary in fitting
    m1.show()
    # print("Parameter 1 name:", m1(34).name)
    # print("Parameter 1 value:", m1(34).values)
    # print("setting values here:")
    m1(34).values = [1.0, 0.01, 0.0, 0.0, 2.0, 2.0]
    # print("were values set?")
    Fit.nIterations = 1000      # Set maximum number of fitting trials to a high number

    # Initialize and fit the chosen model
    if model_choice == "chisoth":
        # m1 = Model("chisoth")
        Fit.perform()
        m1(34).frozen=False
        m1(35).frozen=False
        m1(12).frozen = False
        m1(13).frozen = False
        m1(14).frozen = False
        m1(16).frozen = False
    elif model_choice == "vvapec":
        # s1(1).frozen=False
        # s1(2).frozen=False
        # m1 = Model("vvapec")
        Fit.perform()
        m1(34).frozen=False
        m1(35).frozen=False
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
    AllModels.show(parIDs="1 2 13 14 16 17 31")
    # AllModels.show(parIDs="1 2")

    # Save model configuration to .xcm file
    # fxcm = specfile.replace('.pha', '_Fit.xcm')
    # if os.path.isfile(fxcm):
    #     os.remove(fxcm)
    # Xset.save(fxcm)
    # print("Model saved to", fxcm)
    end_time = time.perf_counter()
    const= m1(34).values[0]
    # const= 1
    print(f"Fitting took {end_time - start_time:.6f} seconds")
    # Open the FITS file, modify the data, and save it
    # output_fits_path=os.path.join(tablefiles,f"t{dt}.fits")
    with fits.open("tbmodel.fits", mode='update') as hdul:
        # Access the data in the primary HDU (usually HDU 0, but can vary)
        data=hdul[3].data
        data["INTPSPEC"] *= const
        # Create a new BinTableHDU with the modified data
        new_bintable_hdu = fits.BinTableHDU(data=data, header=hdul[3].header)
        hdul_new = fits.HDUList([hdul[0], hdul[1],hdul[2],new_bintable_hdu])  # Include primary HDU and modified table HDU
        hdul_new.writeto(tablefile, overwrite=True)
        # Write changes to a new file
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
            f.write(f"{item[0]:.6f}\t{item[1]:.6f}\t{format_value(item[2])}\n")
    
    print(f"Spectrum data saved to {output_filename}")


def format_value(val, threshold=1e-6):
    return f"{val:.6f}" if abs(val) >= threshold else f"{val:.1e}"
# Example usage:
# Set the paths for the input files and the model directory
# dt="20210827"
# process_spectrum("20210827")
# specbase = f"ch2_xsm_{dt}_l1"
# specfile = specbase+'.pha'
# backfile = '/home/heasoft/xsmdas/caldb/CH2xsmresponse20200423v01.rmf'
# backfile = '/home/heasoft/xsm_analysis/scripts/ch2_xsm_20200128_bkg.pha'
# specname=specbase.split("/")[-1]
# Choose the model
# model_choice = "vvapec"  # Options: "chisoth" or "vvapec"
# output_filename = "spectrum/"+specname+f"_{model_choice}_2.txt"
Xset.logChatter = 0
Xset.chatter = 0

# Set up the model
# model_solar(specfile, backfile, modelfile, model_choice=model_choice)
# Calculate and save flux
# calc_flux( output_filename, energy_start=0.1, energy_end=30.0, energy_step=0.01)

import os
import glob
from datetime import datetime
import multiprocessing
lock = multiprocessing.Lock()

def process_files_in_xsm_folder(base_folder="/home/heasoft/data/xsm/xsm_all/xsm/xsm/data"):
    # Define output directories
    spectrum_folder = "spectrum"
    table_folder = os.path.join(spectrum_folder, "table")
    os.makedirs(spectrum_folder, exist_ok=True)
    os.makedirs(table_folder, exist_ok=True)
    tasks = []

    # Traverse through date-structured folders in 'xsm'
    for year_dir in sorted(glob.glob(os.path.join(base_folder, "*/"))):
        year = os.path.basename(os.path.dirname(year_dir))
        print(year)
        for month_dir in sorted(glob.glob(os.path.join(year_dir, "*/"))):
            month = os.path.basename(os.path.dirname(month_dir))
            for day_dir in sorted(glob.glob(os.path.join(month_dir, "*/"))):
                day = os.path.basename(os.path.dirname(day_dir))
                
                # Create the date string `dt`
                dt = f"{year}{month}{day}"
                
                # Define folder paths
                raw_folder = os.path.join(day_dir, "raw")
                calibrated_folder = os.path.join(day_dir, "calibrated")
                
                # Process each file in the calibrated folder
                for file_path in glob.glob(os.path.join(calibrated_folder, "*.pha")):
                    # process_spectrum_file(file_path, dt, spectrum_folder, table_folder)
                    print("file: ",file_path)
                    if os.path.exists(os.path.join(spectrum_folder,f"ch2_xsm_{dt}_l1_vvapec.txt")):
                        continue
                    tasks.append((file_path, dt, spectrum_folder, table_folder))
        # Use multiprocessing to process files in parallel
    print("Total Jobs: ",len(tasks))
    print("Number of parallel tasks:",multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        pool.starmap(process_spectrum_file, tasks)


def process_spectrum_file(file_path, dt, spectrum_folder, table_folder):
    # Define the base name for files
    specbase = f"ch2_xsm_{dt}_l1"
    specfile = specbase + ".pha"
    res=process_spectrum(dt)
    if not res:
        with lock:
            with open("failed.txt","a") as f:
                f.write(file_path+"\n")
        return
    backfile = "/home/heasoft/xsm_analysis/scripts/ch2_xsm_20200128_bkg.pha"
    specname = specbase.split("/")[-1]
    model_choice = "vvapec"  # Use "chisoth" or "vvapec" as needed
    output_filename = os.path.join(spectrum_folder, f"{specname}_{model_choice}.txt")
    
    # Set XSPEC options
    Xset.logChatter = 0
    Xset.chatter = 0
    
    # Run model setup and calculate flux
    modelfile = None  # Assuming this is defined elsewhere
    final_table_file = os.path.join(table_folder, f"t{dt}.fits")
    model_solar(specfile, backfile, final_table_file, model_choice=model_choice)
    calc_flux(output_filename, energy_start=0.1, energy_end=30.0, energy_step=0.01)
    
    # Move and rename files as required
    shutil.move(specfile, os.path.join("spec_pha",specfile))
    shutil.move(specbase+".arf", os.path.join("spec_arf",specbase+".arf"))
    print(f"Processed {file_path} and saved outputs to {output_filename} and {final_table_file}")

# Assuming `process_spectrum`, `model_solar`, and `calc_flux` are defined functions, call the main function:
process_files_in_xsm_folder()
