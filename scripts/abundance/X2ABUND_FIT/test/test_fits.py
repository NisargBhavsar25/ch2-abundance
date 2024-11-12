from astropy.io import fits
import os

file="tbmodel_20210827T210316000_20210827T210332000.fits"
if(file.endswith('.fits')):
    hdul = fits.open("tbmodel_20210827T210316000_20210827T210332000.fits")
    print(hdul[1].data)
    hdul.close()