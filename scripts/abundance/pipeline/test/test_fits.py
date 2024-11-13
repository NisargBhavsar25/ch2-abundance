from astropy.io import fits
import os
import numpy as np

from astropy.io import fits
import numpy as np


# file="/home/heasoft/ch2-abundance/scripts/abundance/test/ch2_cla_l1_20200529T104516257_20200529T104524257.fits"
file="/home/heasoft/ch2-abundance/scripts/abundance/test/converted/ch2_cla_l1_20200529T104516257_20200529T104524257.fits"
# file="/home/heasoft/ch2-abundance/scripts/abundance/pipeline/test/ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits"
if(file.endswith('.fits')):
    hdul = fits.open(file)
    print(hdul[1].columns)
    print(hdul[1].header)
    print(hdul[1].data.shape)
    # data=hdul[3].data
    print(hdul[1].data["CHANNEL"])

    # data["INTPSPEC"] *= 2
    # print(data)
    # Create a new BinTableHDU with the modified data
    # new_bintable_hdu = fits.BinTableHDU(data=data, header=hdul[3].header)
    
    hdul.close()