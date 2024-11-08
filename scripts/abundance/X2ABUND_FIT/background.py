from astropy.io import fits
import os

path = '/home/manasj/isro/code/30'
solarang = []
for file in os.listdir(path):
    if(file.endswith('.fits')):
        hdul = fits.open(os.path.join(path, file))
        solarang.append(hdul[1].header['SOLARANG'])
        print(hdul[0].header)
        hdul.close()

print(solarang)