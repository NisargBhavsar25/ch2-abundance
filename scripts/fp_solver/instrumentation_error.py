import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import gc
import os
import polars as pl
from sklearn.cluster import KMeans
from scipy.special import kl_div
import warnings
import xraylib
from scipy.signal import find_peaks
import pywt
warnings.filterwarnings('ignore')

lazy = pl.scan_parquet('') #import processed files
df = lazy.collect().to_pandas()

def KLine(ele):
  if 'Kb' not in ele:
    if 'Ka' in ele:
      ele = ele[:-2]
    atomic_number = xraylib.SymbolToAtomicNumber(ele)
    ka_line_energy = xraylib.LineEnergy(atomic_number, xraylib.KA1_LINE)
  else:
    ele = ele[:-2]
    atomic_number = xraylib.SymbolToAtomicNumber(ele)
    ka_line_energy = xraylib.LineEnergy(atomic_number, xraylib.KB1_LINE)
  return ka_line_energy

elements = ['O', 'Al', 'Mg', 'Si', 'S', 'KKa', 'CaKa', 'CaKb', 'FeKa', 'FeKb', 'TiKa', 'TiKb', 'CrKa', 'MnKa', 'NiKa', 'NiKb']

klines = [KLine(ele) for ele in elements]

elements = ['O', 'Mg', 'Al', 'Si', 'S', 'KKa', 'CaKa', 'CaKb', 'TiKa', 'TiKb', 'CrKa', 'MnKa', 'FeKa', 'FeKb', 'NiKa', 'NiKb']
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

def classification(x):
  """
  Classification:
  N: No spectrum
  D: No Si detected, other elements
  C: Si detected
  B: Al, Mg, Si
  A: Al, Mg, Si + others
  """
  if len(x) == 0:
    return 'N'
  if 'Si' not in x:
    return 'D'
  if 'Si' in x and 'Al' not in x and 'Mg' not in x:
    return 'C'
  if 'Si' in x and 'Al' in x and 'Mg' not in x:
    return 'B'
  if 'Si' in x and 'Al' in x and 'Mg' in x:
    if len(x) == 3:
      return 'A'
    else:
      return 'X'
  return 'N'

def elements_detection(x, region_indices, elements):
  list_ = set([])
  peaks, _ = find_peaks(x, prominence = 10, width = 3, height = 3)
  peaks2, _ = find_peaks(x, prominence = 9, width = 1, height = 1)

  for peak in peaks:
    for i in range(len(region_indices)):
      if region_indices[i][0] <= peak <= region_indices[i][1]:
        list_.add(elements[i])
        break
  for peak in peaks2:
    for i in range(4, len(region_indices)):
      if region_indices[i][0] <= peak <= region_indices[i][1]:
        list_.add(elements[i])
        break
  return list_
  # return classification(list_)

tqdm.pandas()
df['elements'] = df.progress_apply(lambda row: elements_detection(row['cleaned_dynamic'], region_indices, elements), axis = 1)
instru_err = df[df['elements'] == set(['O'])]
instru_err.to_parquet('') #save file to a parquet