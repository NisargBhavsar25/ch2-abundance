from astropy.io import fits
import os
import numpy as np
file="tbmodel_20210827T210316000_20210827T210332000.fits"
if(file.endswith('.fits')):
    hdul = fits.open("/home/heasoft/xsm_analysis/scripts/xsm_files/table_files/t20210826.fits")
    print(hdul[1].columns)
    print(hdul[2].columns)
    print(hdul[3].columns)
    print(hdul[1].data)
    data=hdul[3].data

    # data["INTPSPEC"] *= 2
    print(data)
    # Create a new BinTableHDU with the modified data
    new_bintable_hdu = fits.BinTableHDU(data=data, header=hdul[3].header)
    
    hdul.close()