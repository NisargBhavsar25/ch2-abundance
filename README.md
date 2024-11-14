# CH2 Abundance Project

This project is focused on analyzing and processing data related to the Chandrayaan-2 mission. The directory structure is organized to facilitate various tasks such as preprocessing, data analysis, and validation.

## Directory Structure

### Root Directory

- **main.py**: The main entry point for the project.
- **README.md**: This file, providing an overview of the project and its structure.
- **requirements.txt**: Lists the dependencies required for the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.

### Preprocess Directory

- **classnames.txt**: Contains class names used in preprocessing.

### Scripts Directory

This directory contains various subdirectories and scripts for different tasks:

- **abundance**: Contains scripts related to abundance analysis.
  - **earthmover**: Subdirectory for Earth Mover's Distance calculations.
  - **optimizer.py**: Script for optimization tasks.
  - **pipeline**: Contains scripts for processing FITS files.
    - **fits_utils.py**: Utility functions for handling FITS files.
    - **mergeFits.py**: Script to merge FITS files.
    - **parallel.py**: Script for parallel processing.
    - **test**: Contains test scripts.
      - **test_fits.py**: Test script for FITS file processing.
  - **X2ABUND_FIT**: Contains scripts for fitting models to data.
    - **mergeFits.py**: Script to merge FITS files.

- **albedo_overlay**: Contains scripts for overlaying albedo data.
  - **xrf_line_lunar_mapping.py**: Script for mapping XRF lines on lunar data.

- **flare_catalog**: Contains scripts for managing flare catalogs.
  - **pipeline.py**: Script to run the flare catalog pipeline.

- **flare_ops**: Contains scripts for flare operations.
  - **scripts**: Subdirectory for flare operation scripts.
    - **solar_model.py**: Script for processing solar model data.

- **mineral_groups**: Contains scripts related to mineral group analysis.

- **subpixelization**: Contains scripts for subpixel analysis.
  - **documentation.md**: Documentation for subpixelization methods.

- **utils**: Contains utility scripts.

- **validation**: Contains scripts for validation tasks.

### Solar Directory

- **fits_names.txt**: List of FITS file names.
- **merged_data.parquet**: Merged data in Parquet format.
- **solar_process.ipynb**: Jupyter notebook for solar data processing.
- **solar.pkl**: Pickle file containing solar data.
- **test.parquet**: Test data in Parquet format.

### Code Files Directory

- **ch2-abundance**: Contains additional scripts and directories.
  - **.gitignore**: Specifies files and directories to be ignored by Git.
  - **knee_point**: Subdirectory for knee point analysis.
  - **README.md**: Additional README file.
  - **scripts**: Contains additional scripts.

### Other Directories

- **Images**: Directory for storing images.
- **Papers**: Directory for storing research papers.

### Other Files

- **tiff.py**: Script for handling TIFF files.
- **validation.py**: Script for validation tasks.
