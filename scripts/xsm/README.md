List of functions to be used for processing XSM files, to extract spectral data and plot it in the form of flux vs energy 

./xsm/extract_spectra/solar_test.py - input the XSM file and output directories, provides the Model Flux in a txt file, fits file, and spectrum

./xsm/xsm_plots - Sample spectrum plots for flares on certain dates, classified according to flare_class_with_bg

./xsm/xsm_scripts/classifyFlarewise.py - Given a csv file with data stored group wise, each group consisting of flares within occurring in a particular region on the moon, classifies all the data group wise as well as according to the particular flare type

./xsm/xsm_scripts/findNearest.py - Calculates the centroid of the latitudes and longitudes, and groups all proximal sets of flares within a certain distance using a Ball Tree.
