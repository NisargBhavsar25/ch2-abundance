import os
import numpy as np
import jax
import jax.numpy as jnp
from astropy.io import fits
from jaxspec.data import ObsConfiguration
from jaxspec.fit import MCMCFitter
from jaxspec.model.background import SubtractedBackground
import xraylib
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V3 import xrf_comp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist

# Read static parameters from file
static_parameter_file = "static_par_localmodel.txt"
try:
    with open(static_parameter_file, "r") as fid:
        finfo_split = fid.read().strip().split('\n')
except FileNotFoundError:
    raise FileNotFoundError(f"Static parameter file '{static_parameter_file}' not found.")

# Static parameters
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])


class JaxPy:
    def __init__(self):
        self.model = None
        self.obs = None
        self.prior = None

    # def preprocess_rmf_arf(self, rmf_file, arf_file):
    #     """Preprocess RMF and ARF files to ensure compatibility."""
    #     try:
    #         with fits.open(rmf_file, mode='update') as hdul:
    #             f_chan_column_pos = 4
    #             tlmin_key = f"TLMIN{f_chan_column_pos}"
    #             for hdu in hdul:
    #                 if hdu.name == 'MATRIX':
    #                     hdu.header[tlmin_key] = 1.00000e-04
    #                 if 'GAIN USE' in hdu.header:
    #                     del hdu.header['GAIN USE']
    #             hdul.flush()

    #         with fits.open(arf_file, mode='update') as hdul:
    #             for hdu in hdul:
    #                 if hasattr(hdu, 'columns'):
    #                     if 'E_LOW' in hdu.columns.names and 'E_HIGH' in hdu.columns.names:
    #                         hdu.columns.change_name('E_LOW', 'ENERG_LO')
    #                         hdu.columns.change_name('E_HIGH', 'ENERG_HI')
    #                 if 'GAIN USE' in hdu.header:
    #                     del hdu.header['GAIN USE']
    #             hdul.flush()
    #     except Exception as e:
    #         raise RuntimeError(f"Error processing RMF/ARF files: {e}")

    def add_spectra(self, data_file, bkg_file):
        self.obs = ObsConfiguration.from_pha_file(
            data_file,
            bkg_path=bkg_file,
            low_energy=0.9,
            high_energy=4.2
        )

    def custom_model(self, params, energy):
        """Custom model to generate the XRF spectrum."""
        print("got into the model")
        energy_mid = (energy[:-1] + energy[1:]) / 2.0

        # Atomic numbers and weights
        at_no = jnp.array([26, 22, 20, 14, 13, 12, 11, 8])  # Fe, Ti, Ca, Si, Al, Mg, Na, O
        weight = jnp.array(params)  # Model parameters

        # Angular and solar parameters
        i_angle = 90.0 - solar_zenith_angle
        e_angle = 90.0 - emiss_angle
        energy_solar, _, counts_solar = jnp.array(np.loadtxt(solar_file, unpack=True))

        # XRF lines
        k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE, xraylib.KM2_LINE, xraylib.KM3_LINE, xraylib.KM4_LINE, xraylib.KM5_LINE])
        l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE, xraylib.L1M3_LINE, xraylib.L1M4_LINE, xraylib.L1M5_LINE, xraylib.L1N1_LINE, xraylib.L1N2_LINE, xraylib.L1N3_LINE, xraylib.L1N4_LINE, xraylib.L1N5_LINE, xraylib.L1N6_LINE, xraylib.L1N7_LINE])
        l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE, xraylib.L2M4_LINE, xraylib.L2M5_LINE, xraylib.L2N1_LINE, xraylib.L2N2_LINE, xraylib.L2N3_LINE, xraylib.L2N4_LINE, xraylib.L2N5_LINE, xraylib.L2N6_LINE, xraylib.L2N7_LINE])
        l3_lines = [xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE, xraylib.L3M4_LINE, xraylib.L3M5_LINE, xraylib.L3N1_LINE,xraylib.L3N2_LINE, xraylib.L3N3_LINE, xraylib.L3N4_LINE, xraylib.L3N5_LINE, xraylib.L3N6_LINE, xraylib.L3N7_LINE]

        xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
        const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
        xrf_struc = xrf_comp(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf)

        # Generate XRF spectrum
        bin_size = energy[1] - energy[0]
        ebin_left = energy_mid - 0.5 * bin_size
        ebin_right = energy_mid + 0.5 * bin_size

        spectrum_xrf = jnp.zeros(len(energy_mid))

        for i in range(len(at_no)):
            for j in range(xrf_lines.lineenergy.shape[1]):
                line_energy = xrf_lines.lineenergy[i, j]
                bin_index = jnp.where((ebin_left <= line_energy) & (ebin_right >= line_energy))[0]
                if bin_index.size > 0:
                    spectrum_xrf = spectrum_xrf.at[bin_index[0]].add(xrf_struc.total_xrf[i, j])

        # Scale spectrum
        scaling_factor = (12.5 * 1e4 * 12.5 * (round(exposure / 8.0) + 1) * 1e4)/(exposure * 4 * jnp.pi * (altitude * 1e4)**2)
        return scaling_factor * spectrum_xrf

    def define_prior(self):
        """Define priors for model parameters."""
        self.prior = {
            f'param_{i+1}': dist.Uniform(0, 100) for i in range(8)  # 8 elements
        }

    def fit(self):
        # class CustomModel:
        #     """Wrapper for the custom model function to conform to the required interface."""
        #     def __init__(self, model_fn):
        #         self.model_fn = model_fn

        #     def photon_flux(self, params, energy):
        #         return self.model_fn(params, energy)

        # Wrap the custom model in the required structure
        # model = CustomModel(self.custom_model)

        # Initialize fitter
        fitter = MCMCFitter(self.model, self.prior, self.obs, background_model=SubtractedBackground())
        result = fitter.fit(num_samples=100, num_chains=3)
        result.plot_ppc()
        plt.show()
        return result



# Initialize and run
jaxpy = JaxPy()
jaxpy.preprocess_rmf_arf("class_rmf_v1.rmf", "class_arf_v1.arf")
jaxpy.add_spectra(
    "ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits",
    "ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits"
)
jaxpy.define_prior()
result = jaxpy.fit()

# Extract best-fit parameters
best_params = result.samples.mean(axis=0)

# Generate model spectrum for plotting
energy = jnp.linspace(0.9, 4.2, 1000)
model_spectrum = jaxpy.custom_model(best_params, energy)

# Plot the results
plt.figure(figsize=(10, 6))
plt.step(jaxpy.obs.energies, jaxpy.obs.counts, where="mid", label="Observed Data")
plt.plot(energy, model_spectrum, label="Best-fit Model", color='red')
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.title("XRF Spectral Fitting")
plt.legend()
plt.show()
