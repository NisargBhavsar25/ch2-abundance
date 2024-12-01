# ISRO : High-Resolution Elemental Mapping of Lunar Surface

<!-- TABLE OF CONTENTS -->
<details>
    <summary>Table of Contents</summary>
    <ol>
        <li><a href="#about-the-project">About The Project</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
        <li><a href="#installation">Installation</a></li>
    </ol>
</details>

<!-- TOC --><a name="about-the-project"></a>
## About the project

This project uses data from Chandrayaan-2’s CLASS instrument to make detailed maps of the Moon’s surface. These maps show the different elements present on the Moon. By studying the intensity of X-ray fluorescence (XRF) lines emitted by elements like magnesium, Silicon, and Aluminum, induced by solar flares, and calculating ratios such as Mg/Si and Al/Si, the project derives spatially resolved compositional information while mitigating the influence of varying solar flare conditions. A strong data processing pipeline extracts, models, and validates elemental abundances from spectral data, facilitating the creation of compositional maps with a spatial resolution of approximately 12 km. These maps, overlaid on lunar albedo data, illuminate compositional variations, providing valuable insights into mineral distribution and potential in-situ resource locations. This research contributes to lunar exploration efforts by deepening our geological understanding and aiding in future mission planning.

<!-- TOC --><a name="project-structure"></a>
## Project Structure
```
.
├── main.py
├── preprocess
├── scripts
│   ├── abundance
│   │   ├── earthmover
│   │   │   └── earthmover.py
│   │   ├── optimizer.py
│   │   ├── pipeline
│   │   │   ├── get_xrf_lines_V2.py
│   │   │   └── xrf_comp_new_V3.py
│   │   └── x2abund.py
│   ├── albedo_overlay
│   │   └── xrf_line_lunar_mapping.py
│   ├── flare_catalog
│   │   └── pipeline.py
│   ├── flare_ops
│   │   └── scripts
│   │       └── solar_model.py
│   ├── mineral_groups
│   │   └── mineral.py
│   ├── subpixelization
│   │   └── gaussian_avg.py
│   ├── utils
└── solar
    └── solar_process.ipynb
```

<!-- TOC --><a name="installation"></a>
## Installation

```bash
pip install -r requirements.txt
```