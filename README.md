# CH2 Abundance Project

This project is focused on analyzing and processing data related to the Chandrayaan-2 mission. The directory structure is organized to facilitate various tasks such as preprocessing, data analysis, and validation.

## Directory Details

### Root Directory

- **.gitignore**: Specifies files and directories for Git to ignore.
- **main.py**: The primary script to run the project's main functionalities.
- **README.md**: Contains the project's overview and detailed directory structure.
- **requirements.txt**: Lists all Python dependencies required for the project.

### preprocess/

- **classnames.txt**: A text file listing class names used during data preprocessing.

### scripts/

Contains various scripts organized into subdirectories based on their functionality.

#### abundance/

Focuses on abundance analysis.

- **earthmover/**: Scripts for calculating Earth Mover's Distance.
  - **emd_calc.py**: Computes the Earth Mover's Distance between distributions.
- **optimizer.py**: Implements optimization algorithms for abundance estimation.
- **pipeline/**: Scripts forming the data processing pipeline.
  - **fits_utils.py**: Utility functions for handling FITS files.
  - **mergeFits.py**: Merges multiple FITS files into one.
  - **parallel.py**: Enables parallel processing to speed up computations.
  - **test/**: Contains unit tests for pipeline scripts.
    - **test_fits.py**: Tests FITS file processing functions.
- **X2ABUND_FIT/**: Scripts for fitting models to data for abundance calculations.
  - **fit_model.py**: Fits statistical models to the processed data.

#### albedo_overlay/

Scripts for overlaying albedo data onto lunar maps.

- **xrf_line_lunar_mapping.py**: Maps XRF spectral lines onto lunar surface data.

#### flare_catalog/

Manages and processes flare catalog data.

- **pipeline.py**: Executes the flare catalog data processing pipeline.
- **utils.py**: Utility functions specific to flare catalog processing.

#### flare_ops/

Handles flare operations.

- **scripts/**:
  - **solar_model.py**: Processes and analyzes solar model data.
  - **flare_analysis.py**: Performs analysis on flare data.

#### mineral_groups/

Analyzes different mineral groups identified on the lunar surface.

- **mineral_analysis.py**: Identifies and analyzes mineral compositions.

#### subpixelization/

Deals with subpixel analysis techniques.

- **subpixel.py**: Implements algorithms for subpixel calculations.
- **documentation.md**: Provides detailed explanations of subpixelization methods used.

#### utils/

Contains utility scripts used across various parts of the project.

- **file_utils.py**: Functions for file reading, writing, and handling.
- **math_utils.py**: Common mathematical functions and constants.
- **plotting_utils.py**: Functions for creating plots and visualizations.

#### validation/

Includes scripts dedicated to validating data and results.

- **validate_data.py**: Validates datasets for consistency and correctness.
- **validation_report.md**: Reports generated from validation processes.

### solar/

Dedicated to solar data processing.

- **fits_names.txt**: Lists the names of all FITS files used in the project.
- **merged_data.parquet**: Contains merged solar data in Parquet format.
- **solar_process.ipynb**: Jupyter Notebook for processing solar data.
- **solar.pkl**: Pickled object of processed solar data for quick loading.
- **test.parquet**: Parquet file with test data for validating solar data processing.

### Code_files/

Contains additional code and scripts related to the project.

- **ch2-abundance/**:
  - **.gitignore**: Git ignore file for this subdirectory.
  - **knee_point/**: Scripts for identifying knee points in data analysis.
    - **find_knee.py**: Algorithm to detect knee points in curves.
  - **README.md**: Additional documentation for code files.
  - **scripts/**:
    - **helper_scripts.py**: Supplementary helper functions.

### Images/

Stores images related to the project, such as figures, graphs, and diagrams.

- **lunar_map.png**: High-resolution image of the lunar surface map.
- **abundance_plot.png**: Graphical representation of abundance data.

### Papers/

Contains research papers, articles, and other reference materials supporting the project.

- **paper1.pdf**: Research on lunar mineralogy relevant to the project.
- **paper2.pdf**: Studies on data analysis methods used in this project.

### Other Files

- **tiff.py**: Script for processing and converting TIFF image files.
- **validation.py**: Contains functions and scripts for validating datasets and results.

---

This `README.md` provides a detailed overview of the project's directory structure, including all files and directories with brief descriptions to enhance clarity and understanding.
