# ISRO : High-Resolution Elemental Mapping of Lunar Surface

<!-- TABLE OF CONTENTS -->
<details>
    <summary>Table of Contents</summary>
    <ol>
        <li><a href="#about-the-project">About The Project</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usages</a></li>

    </ol>
</details>

<!-- TOC --><a name="about-the-project"></a>
## About the project

This project uses data from Chandrayaan-2’s CLASS instrument to make detailed maps of the Moon’s surface. These maps show the different elements present on the Moon. By studying the intensity of X-ray fluorescence (XRF) lines emitted by elements like magnesium, Silicon, and Aluminum, induced by solar flares, and calculating ratios such as Mg/Si and Al/Si, the project derives spatially resolved compositional information while mitigating the influence of varying solar flare conditions. A strong data processing pipeline extracts, models, and validates elemental abundances from spectral data, facilitating the creation of compositional maps with a spatial resolution of approximately 12 km. These maps, overlaid on lunar albedo data, illuminate compositional variations, providing valuable insights into mineral distribution and potential in-situ resource locations. This research contributes to lunar exploration efforts by deepening our geological understanding and aiding in future mission planning.

<!-- TOC --><a name="project-structure"></a>
## Project Structure
```
C:.
│   main.py
│   README.md
│   requirements.txt
│
├── preprocess
│
├── scripts
│   ├── __init__.py
│   ├── abundance
│   │   └── pipeline
│   │       ├── common_modules.py
│   │       ├── define_xrf_localmodel.py
│   │       ├── fits_utils.py
│   │       ├── fit_optuna.py
│   │       ├── get_constants_xrf_new_V2.py
│   │       ├── get_xrf_lines_V1.py
│   │       ├── get_xrf_lines_V2.py
│   │       ├── parallel.py
│   │       ├── xrf_comp_new_V2.py
│   │       ├── xrf_comp_new_V3.py
│   │       │
│   │       ├── BKG
│   │       ├── data_constants
│   │       │   ├── atomicweight.txt
│   │       │   ├── fluores_yield.txt
│   │       │   ├── fnt_frel.txt
│   │       │   ├── kalpha_be_density_kbeta.txt
│   │       │   └── readme.txt
│   │       │
│   │       └── test
│   │           ├── ch2_xsm_20191128_bkg.pha
│   │           ├── class_arf_v1.arf
│   │           ├── class_rmf_v1.rmf
│   │           ├── test_fits.py
│   │           ├── test_x2abund.py
│   │           └── xcm_x2abund_test.xcm
│   │
│   └── test
│       └── converted
│
├── albedo_overlay
│   └── xrf_line_lunar_mapping.py
│
├── flare_catalog
│   ├── classifier.py
│   ├── classifier_loop.py
│   ├── pipeline.py
│   ├── pipeline.sh
│   └── earthmover
│       ├── earthmover.py
│       └── utils.py
│
├── flare_ops
│   └── scripts
│       ├── solar_model.py
│       ├── tbmodel.fits
│       └── spectrum
│           ├── table
│           ├── spec_arf
│           └── spec_pha
│
├── mineral_groups
│   └── find_rocks.py
│
├── subpixelization
│   └── gaussian_avg.py
│
└── utils
    └── knee_point.py

```

<!-- TOC --><a name="installation"></a>
## Installation

```bash
pip install -r requirements.txt
```
## Usage
#### Gaussian Documentation
The purpose of this implementation is to simulate spatial elemental abundances on a grid using Gaussian-distributed values within defined geographic boundaries. This model is particularly suited for applications in planetary surface mapping, geochemical studies, and environmental monitoring where it is essential to interpolate sparse data points across a defined region. The Gaussian-based interpolation technique herein provides smooth transitions across regions, allowing for a more accurate portrayal of elemental distributions as seen in natural settings, such as lunar surface.
It is well suited for reconstructing abundance maps fromelemental fluorescence data, such as those collected by X-rayspectrometry instruments like the Chandrayaan-2 CLASS. Suchspectrometers obtain X-ray fluorescence data from a specificarea at once(12.5km x 12.5km for CLASS),  representingconcentrations of elements like oxygen, magnesium, aluminum,and silicon.

Code Structure and Components
Class GaussianArray
The main class in this implementation, GaussianArray, creates a 2D spatial grid populated with Gaussian values based on input coordinates. The class comprises several key methods:
- **__init__**: Initializes a zero-valued 3D array (x, y, 2) to store both abundance values and a count tracker of number of overlaps for each cell in the grid.
- **in_block_or_not**: Checks if the input coordinates (latitude and longitude) for a polygonal region fall within the specified block, defined by a bounding box of latitude and longitude values.
generate_gaussian_distribution: Generates a normalized 2D Gaussian distribution over the desired region. This Gaussian distribution is used to represent the concentration of elements at a given location, with sigma parameter controlling the spread. 
- **fill_up_the_array**: Populates the array with Gaussian values based on a polygon defined by input coordinates. The Gaussian values are confined within the polygon using a mask, created using a polygon-drawing algorithm. This method ensures that interpolated abundance values are restricted to the designated region of interest.
---
Usage 
In the example code provided, multiple regions are defined with their coordinates, bounding boxes, and abundance parameters (i.e., maximum value and sigma). By calling add_gaussian_box for each region, the model iteratively updates the grid with interpolated abundance values, visualizing each addition on demand.
```Python
gaussian_array = GaussianArray(grid_size=64)
gaussian_array.add_gaussian_box([-20, 20, 25, -30], [-20, -10, 30, 25], [-100, 100, 100, -100], [-100, -100, 100, 100], 1, sigma=5, plot=True)
```
By repeating this process, the class allows for multiple Gaussian distributions to be mapped onto the grid, simulating the combined intensity across spatial regions.
Setting the Sigma Parameter
The sigma parameter (σ) in Gaussian distribution is crucial for tuning the spread of values across the grid. In elemental abundance applications, sigma should reflect the resolution of the input data or the desired smoothness level. For example, in planetary mapping, a sigma matching the grid resolution (e.g., 12.5 km, as in Chandrayaan-2 CLASS) would maintain fidelity to observed data while avoiding excessive smoothing that could obscure regional variations.
Gaussian Method Justification
The Gaussian method provides several benefits for this use case:
- Smoothing Sparse Data: Gaussian interpolation naturally handles sparse data points by diffusing values across neighboring cells, reducing abrupt changes in abundance.
- Naturalistic Distribution Modeling: The bell curve of Gaussian distribution effectively represents concentrations of elements in a region, modeling elemental decay outward from a central point of interest. Also simulates the instrument’s focus area as the central region contributes more to the collected data.
- Computational Efficiency: The use of Gaussian functions within defined polygons is computationally efficient, especially with vectorized operations in NumPy, and offers quick visualization updates.
