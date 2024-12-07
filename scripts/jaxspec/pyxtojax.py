# convert parallel.py to jaxspec - make a class with functions

# note - u cant modify jnp arrays, it will return a new array always when operation takes place

#imports

import matplotlib.pyplot as plt
import numpyro.distributions as dist
import os
import pdb
from astropy.io import fits
import jax.numpy as jnp
import jax
from jaxspec.data import ObsConfiguration
from jaxspec.fit import MCMCFitter
from jaxspec.model.background import SubtractedBackground
import haiku as hk
import numpy as np
import xraylib
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V3 import xrf_comp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from jaxspec.model.abc import AdditiveComponent


static_parameter_file = "static_par_localmodel.txt"
fid = open(static_parameter_file,"r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

els= ['fe','ti','ca','si','al','mg','na','o']

# in the fits header
# RESPFILE= 'class_rmf_v1.rmf'   / associated redistrib matrix filename
# ANCRFILE= 'class_arf_v1.arf'   / associated ancillary response filename
from customModels import CustomAdditiveComponent

class CustomModel(CustomAdditiveComponent):
    # photon_flux = None
    # def continuum(self,energy):
    #     pass
    def emission_lines(self, e_lo,e_hi):
        import pdb
        # pdb.set_trace()
        energy=e_lo
        print(energy,len(energy),type(energy))
        energy_mid = np.zeros(np.size(energy)-1)
        for i in np.arange(np.size(energy)-1):
            energy_mid[i] = 0.5*(energy[i+1] + energy[i])
            
        # Defining some input parameters required for x2abund xrf computation modules
        at_no = np.array([26,22,20,14,13,12,11,8])
        weight =[hk.get_parameter(f"h_{el}", shape=(), init=jnp.ones) for el in els]
        # weight = params

        i_angle = 90.0 - solar_zenith_angle
        e_angle = 90.0 - emiss_angle

        (energy_solar,tmp1_solar,counts_solar) = readcol(solar_file,format='F,F,F')
        # this tmp1_solar is not being used
        
        # Computing the XRF line intensities
        k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE, xraylib.KM2_LINE, xraylib.KM3_LINE, xraylib.KM4_LINE, xraylib.KM5_LINE])
        l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE, xraylib.L1M3_LINE, xraylib.L1M4_LINE, xraylib.L1M5_LINE, xraylib.L1N1_LINE, xraylib.L1N2_LINE, xraylib.L1N3_LINE, xraylib.L1N4_LINE, xraylib.L1N5_LINE, xraylib.L1N6_LINE, xraylib.L1N7_LINE])
        l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE, xraylib.L2M4_LINE, xraylib.L2M5_LINE, xraylib.L2N1_LINE, xraylib.L2N2_LINE, xraylib.L2N3_LINE, xraylib.L2N4_LINE, xraylib.L2N5_LINE, xraylib.L2N6_LINE, xraylib.L2N7_LINE])
        l3_lines = [xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE, xraylib.L3M4_LINE, xraylib.L3M5_LINE, xraylib.L3N1_LINE,xraylib.L3N2_LINE, xraylib.L3N3_LINE, xraylib.L3N4_LINE, xraylib.L3N5_LINE, xraylib.L3N6_LINE, xraylib.L3N7_LINE]

        xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
        const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
        xrf_struc = xrf_comp(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf)

        # Generating XRF spectrum
        bin_size = energy[1] - energy[0]
        ebin_left = energy_mid - 0.5*bin_size
        ebin_right = energy_mid + 0.5*bin_size

        no_elements = (np.shape(xrf_lines.lineenergy))[0]
        n_lines = (np.shape(xrf_lines.lineenergy))[1]
        n_ebins = np.size(energy_mid)
        spectrum_xrf = dblarr(n_ebins)
        for i in range(0, no_elements):
            for j in range(0, n_lines):
                line_energy = xrf_lines.lineenergy[i,j]
                bin_index = np.where((ebin_left <= line_energy) & (ebin_right >= line_energy))
                spectrum_xrf[bin_index] = spectrum_xrf[bin_index] + xrf_struc.total_xrf[i,j]
        
        # Defining the flux array required for XSPEC
        scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)
        spectrum_xrf_scaled = scaling_factor*spectrum_xrf
        
        flux=[0]*1024
        for i in range(0, n_ebins):
            flux[i] = spectrum_xrf_scaled[i] 
        import pdb
        # pdb.set_trace()
        # print(len(flux),len(energy))
        flux = jnp.array(flux)
        return flux, (e_lo+e_hi)/2

model = CustomModel()

class JaxPy():
    def __init__(self):
        self.model = None
        self.obs = None
        self.prior = {f"h_{el}":dist.Normal(0.0, 1) for el in els}

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
    
    # the parameters that are to be adjusted to fit are in parallel.py
    def fit_jax(self):
        # pdb.set_trace()
        key = jax.random.PRNGKey(0)  # Add this line to create a random key
        import pdb
        pdb.set_trace()
        fitter = MCMCFitter(model, self.prior, self.obs, background_model=SubtractedBackground())
        result = fitter.fit(num_samples=100, num_chains=3,rng_key=key)
        result.plot_ppc()
        plt.show();


jaxpy = JaxPy()

# Preprocess the RMF and ARF files
# rmf_file = "class_rmf_v1.rmf"
# arf_file = "class_arf_v1.arf"
# jaxpy.preprocess_rmf_arf(rmf_file, arf_file)

# Load the observation data
data_file = "ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits"
bkg_file = "ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits"
jaxpy.add_spectra(data_file, bkg_file)
result = jaxpy.fit_jax()

