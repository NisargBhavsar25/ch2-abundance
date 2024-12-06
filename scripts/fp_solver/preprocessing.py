import numpy as np
import warnings
import xraylib
import pywt
warnings.filterwarnings('ignore')

file_name = '' # Add file name here

elements = ['O', 'Al', 'Mg', 'Si', 'S', 'KKa', 'CaKa', 'CaKb', 'FeKa', 'FeKb', 'TiKa', 'TiKb', 'CrKa', 'MnKa', 'NiKa', 'NiKb']
CONV_FACTOR = 0.0135
regions = []
for ele in elements:
  if 'Kb' not in ele:
    if 'Ka' in ele:
      ele = ele[:-2]
    atomic_number = xraylib.SymbolToAtomicNumber(ele)
    ka_line_energy = xraylib.LineEnergy(atomic_number, xraylib.KA1_LINE)
  else:
    ele = ele[:-2]
    atomic_number = xraylib.SymbolToAtomicNumber(ele)
    ka_line_energy = xraylib.LineEnergy(atomic_number, xraylib.KB1_LINE)
  lo = ka_line_energy - 0.1
  hi = ka_line_energy + 0.1
  regions.append([lo, hi])
region_indices = []
for r in regions:
  ind = []
  for idx in r:
    ind.append((int)(idx / CONV_FACTOR))
  region_indices.append(ind)
window_size = 10
window_size_roi = 3

kernel = np.ones(window_size) / window_size
kernel_roi = np.ones(window_size_roi) / window_size_roi

def smoothen(x, region_indices=region_indices, dynamic = True):
  global window_size
  global window_size_roi

  y = np.convolve(x, kernel, mode = 'same')
  if not dynamic:
    return y

  for regions in region_indices:
    lo = regions[0]
    hi = regions[1]
    y[lo:hi + 1] = np.convolve(x[lo:hi + 1], kernel_roi, mode = 'same')
  del x
  return y

def clean_signal(signal, region_indices=region_indices, dynamic = True):
  wavelet = 'haar'  # You can choose other wavelets like 'db1', 'db4', etc.
  coeffs = pywt.wavedec(signal, wavelet)
  threshold = 5 # Set a threshold value
  coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

  # Reconstruct the signal from the thresholded coefficients
  denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
  return smoothen(denoised_signal, region_indices, dynamic)

# param_all_files_lazy = pl.scan_parquet(file_name) 
# param_all_files_lazy = param_all_files_lazy.select(['file', 'y'])
# df = param_all_files_lazy.collect().to_pandas()

# tqdm.pandas()
# df["cleaned_dynamic"] = df.progress_apply(lambda row: clean_signal(row['y'].copy(), region_indices, dynamic = True), axis = 1)
# df_new = df[['file', 'cleaned_dynamic']]