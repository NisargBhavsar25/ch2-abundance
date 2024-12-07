# ISRO : High-Resolution Elemental Mapping of Lunar Surface

<!-- TABLE OF CONTENTS -->
## Table of Contents

1. [About The Project](#about-the-project)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Run](#run)
5. [Explanation](#explanation)

<!-- TOC --><a name="about-the-project"></a>
## About the project

This project uses data from Chandrayaan-2’s CLASS instrument to make detailed maps of the Moon’s surface. These maps show the different elements present on the Moon. By studying the intensity of X-ray fluorescence (XRF) lines emitted by elements like magnesium, Silicon, and Aluminum, induced by solar flares, and calculating ratios such as Mg/Si and Al/Si, the project derives spatially resolved compositional information while mitigating the influence of varying solar flare conditions. A strong data processing pipeline extracts, models, and validates elemental abundances from spectral data, facilitating the creation of compositional maps with a spatial resolution of approximately 12 km. These maps, overlaid on lunar albedo data, illuminate compositional variations, providing valuable insights into mineral distribution and potential in-situ resource locations. This research contributes to lunar exploration efforts by deepening our geological understanding and aiding in future mission planning.

<!-- TOC --><a name="project-structure"></a>
## Project Structure
```
.
├── README.md
├── main.py
└── scripts
    ├── abundance
    │   ├── X2ABUND_FIT
    │   │   ├── background.py
    │   │   ├── common_modules.py
    │   │   ├── data_constants
    │   │   ├── fit_larch.py
    │   │   ├── fit_optuna.py
    │   │   ├── get_constants_xrf_new_V2.py
    │   │   ├── get_xrf_lines_V1.py
    │   │   ├── get_xrf_lines_V2.py
    │   │   ├── mergeFits.py
    │   │   ├── parallel.py
    │   │   ├── responsefiles.py
    │   │   ├── scripts
    │   │   │   ├── plot_fits.py
    │   │   │   └── test_fits.py
    │   │   ├── single_fit_optuna.py
    │   │   ├── tempCodeRunnerFile.py
    │   │   ├── tes.py
    │   │   ├── test
    │   │   │   ├── plots_x2abund_test.pdf
    │   │   │   ├── test_fits.py
    │   │   │   ├── test_x2abund.py
    │   │   │   └── xcm_x2abund_test.xcm
    │   │   ├── test.py
    │   │   ├── tif.py
    │   │   ├── xrf_comp_new_V2.py
    │   │   └── xrf_comp_new_V3.py
    │   ├── earthmover
    │   │   ├── earthmover.py
    │   │   └── utils.py
    │   ├── pipeline
    │   │   ├── BKG
    │   │   ├── abundances_results_final.csv
    │   │   ├── common_modules.py
    │   │   ├── data_constants
    │   │   │   ├── ffast
    │   │   │   ├── form_factors
    │   │   │   └── xcom
    │   │   ├── define_xrf_localmodel.py
    │   │   ├── fits_utils.py
    │   │   ├── get_constants_xrf_new_V2.py
    │   │   ├── get_xrf_lines_V1.py
    │   │   ├── get_xrf_lines_V2.py
    │   │   ├── gpufit.py
    │   │   ├── jax_fit.py
    │   │   ├── mergeFits.py
    │   │   ├── merged_data.parquet
    │   │   ├── parallel.py
    │   │   ├── parallelSpec.py
    ├── ball_tree
    │   ├── ball_tree.pkl
    │   └── clustering.py
    ├── fits_utils
    │   ├── addPHAs.py
    │   ├── conv.py
    │   ├── group.py
    │   ├── merge.py
    │   ├── mergesimilar.py
    │   └── phaviz.py
    ├── flare_ops
    │   ├── README.md
    │   ├── flare_catalog
    │   │   ├── classifier.py
    │   │   ├── classifier_loop.py
    │   │   ├── pipeline.py
    │   │   └── pipeline.sh
    │   ├── gen_files.py
    │   ├── getFitsforXSM.py
    │   ├── solar_model.py
    │   ├── solar_model_1.py
    │   ├── solar_test.py
    │   ├── spec_arf
    │   ├── spec_pha
    │   └── spectrum
    ├── fp_solver
    │   ├── __init__.py
    │   ├── claisse_quintin.py
    │   ├── fit.py
    │   ├── intensity_finder.py
    │   ├── main.ipynb
    │   └── preprocessing.py
    ├── image-augmentation
    │   └── Image Augmentation.py
    ├── jaxspec
    │   ├── common_modules.py
    │   ├── customModels.py
    │   ├── data_constants
    │   │   ├── ffast
    │   │   ├── form_factors
    │   │   └── xcom
    │   ├── define_xrf_localmodel.py
    │   ├── get_constants_xrf_new_V2.py
    │   ├── get_xrf_lines_V1.py
    │   ├── parallel.py
    │   ├── pyxtojax.py
    │   ├── test
    │   │   ├── test_fits.py
    │   │   ├── test_x2abund.py
    │   │   └── xcm_x2abund_test.xcm
    │   ├── test3.py
    │   └── xrf_comp_new_V3.py
    ├── map_making
    │   └── gaussian_avg.py
    ├── mineral_groups
    │   └── mineral.py
    ├── monte_carlo
    │   ├── ele_abund_lpgrs.py
    │   ├── images
    │   ├── lpgrs_elemental_maps
    │   │   ├── Al_image.jpeg
    │   │   └── O_grey_scale.jpeg
    │   ├── subprocess_cli.py
    │   ├── sumcounts.py
    │   └── txt_xsmi.py
    ├── solar
    │   ├── merged_data.parquet
    │   ├── solar.pkl
    │   └── solar_process.ipynb
    ├── super-resolution
    │   ├── UNet generalized Flexible.ipynb
    │   ├── Unet
    │   │   ├── dataset.py
    │   │   ├── train.py
    │   │   └── unet-inference.ipynb
    │   ├── elemantal-map-enhancement.ipynb
    │   └── outputs
    ├── utils
    │   ├── IIRS_find_overlaps.py
    │   ├── catalog.py
    │   ├── get-pip.py
    │   ├── high_confidence_ca_al.csv
    │   ├── improve_overlap.py
    │   ├── knee_point.py
    │   ├── lunar_map.py
    │   ├── overlap.py
    │   ├── ratio_to_mineral.py
    │   └── util_utc2met.py
    ├── validation
    │   ├── Elevation_Mapping
    │   │   ├── Elevation_Mapping.py
    │   │   └── Elevation_Mapping_TIFF.py
    │   ├── LPGRS
    │   │   ├── add_header.py
    │   │   ├── images
    │   │   ├── search_for_lat_lon.py
    │   │   └── tab_to_csv.py
    │   ├── TIF
    │   │   ├── TIF_DATA
    │   │   └── tiff.py
    │   └── iirs
    │       ├── extract_lat_lon.py
    │       └── plot_iirs.py
    ├── xrf_fit
    │   ├── GAfit.py
    │   ├── GAfit_plain.py
    │   ├── data_handler.py
    │   ├── data_handler_nobkg.py
    │   ├── element_handler_sub.py
    │   ├── element_model_sub.py
    │   ├── gauss.py
    │   ├── high_confidence_ca_al.csv
    │   ├── optimizer.py
    │   ├── phy.py
    │   ├── phyfit_sub.py
    │   ├── scipy_fit.py
    │   ├── scipy_fit_nobkg copy.py
    │   ├── scipy_fit_nobkg.py
    │   └── spectral_optimizer.py
    └── xsm
        ├── extract_spectra
        │   └── solar_test.py
        ├── xsm_plots
        └── xsm_scripts
            ├── classifyFlarewise.py
            └── findNearest.py

```

<!-- TOC --><a name="installation"></a>
## Installation

```bash
pip install -r requirements.txt
```

<!-- TOC --><a name="run"></a>
## Run
```bash
python main.py
```

<!-- TOC --><a name="explanation"></a>
## Explanation
A simple overview of the modules used in the project, present in ./scripts directory.

**abundance/**  
This directory encompasses a mineral abundance calculation pipeline with multiple analytical components. The implementation integrates Earth Mover's Distance calculations for distribution analysis, alongside robust chi-square minimization algorithms within X2ABUND_FIT. The framework incorporates comprehensive validation protocols to ensure calculation accuracy.

**ball_tree/**  
Implements sophisticated clustering methodologies utilizing ball tree data structures, optimized for high-dimensional spectral data analysis. The implementation includes automated classification mechanisms and persistent model storage capabilities, ensuring efficient data organization and retrieval.

**fits_utils/**  
Houses a comprehensive suite of tools designed for the manipulation and analysis of FITS (Flexible Image Transport System) files. The utilities facilitate PHA data integration, spectral file consolidation, and systematic file organization based on observational parameters, complemented by advanced visualization capabilities.

**flare_ops/**  
Facilitates automated solar flare analysis through sophisticated detection algorithms. The implementation encompasses precise parameter calculations including peak flux determination, duration assessment, and energy quantification, with integrated classification and logging systems.

**fp_solver/**  
Presents a mathematical framework implementing iterative fixed-point methodologies for spectral deconvolution. The system incorporates advanced optimization techniques to handle convergence challenges in complex spectral fitting scenarios.

**image-augmentation/**  
Delivers advanced image enhancement capabilities specifically tailored for spectral imagery. The implementation includes sophisticated noise reduction algorithms, contrast enhancement methodologies, and specialized preprocessing routines for X-ray imagery.

**jaxspec/**   
Presents a high-performance spectral analysis framework leveraging JAX acceleration. The implementation facilitates parallel processing capabilities and GPU-optimized routines for efficient large-scale spectral data processing.

**map_making/**  
Facilitates the creation of high-resolution elemental and mineral distribution maps from spectral data. Incorporates sophisticated interpolation algorithms and coordinate mapping systems for generating publication-grade visualizations.

**mineral_groups/**  
Implements advanced spectral analysis algorithms for mineral classification. The system integrates comprehensive mineral libraries and sophisticated comparison protocols for accurate mineral identification.

**monte_carlo/**  
Provides robust error estimation capabilities through Monte Carlo simulations. The implementation includes uncertainty quantification and error propagation analysis for abundance calculations. It includes a subprocess program to generate xmsi files and then also generate xmso files.

**solar/**  
Facilitates comprehensive solar X-ray data analysis, incorporating advanced algorithms for flare detection, background radiation assessment, and temporal spectral evolution analysis. The system provides specialized routines for processing solar observation time series.

**super-resolution/**  
Implements state-of-the-art deep learning models for spectral image enhancement. The framework incorporates multiple super-resolution algorithms specifically optimized for X-ray and spectral data characteristics.

**utils/**  
Provides essential analytical tools including advanced knee point detection algorithms, peak identification systems, and spectral smoothing functions. The implementation includes comprehensive mathematical operations and sophisticated file handling protocols for diverse data formats.

**validation/**  
Implements rigorous quality assurance protocols, incorporating comprehensive model validation mechanisms and systematic result verification methodologies. The framework includes sophisticated benchmarking systems for evaluating abundance calculations and spectral fitting accuracy.

**xrf_fit/**  
Delivers specialized X-ray fluorescence analysis capabilities, incorporating advanced peak deconvolution algorithms, background correction methodologies, and elemental identification systems. The implementation includes sophisticated calibration protocols for quantitative analysis.

**xsm/**  
Facilitates comprehensive X-ray spectrometer data processing through sophisticated calibration algorithms, background elimination protocols, and response matrix calculations. The system incorporates instrument-specific corrections and standardized data formatting procedures.

