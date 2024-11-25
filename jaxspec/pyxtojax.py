# convert parallel.py to jaxspec - make a class with functions

#imports

import matplotlib.pyplot as plt
import numpyro.distributions as dist
import os
from astropy.io import fits
import jax.numpy as jnp
from jaxspec.data import ObsConfiguration
from jaxspec.fit import MCMCFitter
from jaxspec.model.background import SubtractedBackground

# in the fits header
# RESPFILE= 'class_rmf_v1.rmf'   / associated redistrib matrix filename
# ANCRFILE= 'class_arf_v1.arf'   / associated ancillary response filename

class JaxPy():
    def __init__(self):
        self.model = None
        self.obs = None
        self.prior = None

    def preprocess_rmf_arf(self, rmf_file, arf_file):
        with fits.open(rmf_file, mode='update') as hdul:    # need to execute this once then remove from this class
            # add the TLMIN in the response_file
            f_chan_column_pos = 4
            tlmin_key = f"TLMIN{f_chan_column_pos}"
            for hdu in hdul:
                if hdu.name == 'MATRIX':
                    hdr = hdu.header
                    hdr[tlmin_key] = 1.00000e-04
            # remove the illegal keyword        - need to look on this gain part
            for hdu in hdul:
                hdr = hdu.header
                if 'GAIN USE' in hdr:
                    del hdr['GAIN USE']

            hdul.flush()
        
        # rename columns of the arf file
        with fits.open(arf_file, mode='update') as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'columns'):
                    if 'E_LOW' in hdu.columns.names and 'E_HIGH' in hdu.columns.names:
                        hdu.columns.change_name('E_LOW', 'ENERG_LO')
                        hdu.columns.change_name('E_HIGH', 'ENERG_HI')
            for hdu in hdul:
                hdr = hdu.header
                if 'GAIN USE' in hdr:
                    del hdr['GAIN USE']

            hdul.flush()
    
    def add_spectra(self, data_file, bkg_file):
        obs_config = ObsConfiguration.from_pha_file(
        data_file,
        bkg_path= bkg_file,
        low_energy= 0.9, 
        high_energy= 4.2
        )

        self.obs = obs_config

    def custom_model(self):
        # create self.model and self.prior
        pass

    def fit(self):
        fitter = MCMCFitter(self.model, self.prior, self.obs, background_model=SubtractedBackground())
        result = fitter.fit(num_samples=100, num_chains=3)


        