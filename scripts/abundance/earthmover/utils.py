from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import os
import xml.etree.ElementTree as ET
from scipy.optimize import curve_fit
from scipy.stats import chi2
import pandas as pd
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.signal import savgol_filter

def get_fits_header(fits_data):
  hdul = fits.open(fits_data)
  return hdul[0].header, hdul[1].header

def get_x_y(fits_data):
  hdul = fits.open(fits_data)
  record_array = hdul[1].data.view(np.recarray)
  x, y = record_array.CHANNEL, record_array.COUNTS
  x = np.array(x, dtype = 'float64')
  x = x * 0.0135
  y = np.array(y, dtype = 'float64')
  hdul.close()
  return x, y

def local_solar_timestamp(fits_data):
  header = get_fits_header(fits_data)[1]
  hr = header["LST_HR"]
  min = header["LST_MIN"]
  sec = header["LST_SEC"]
  start_time = ''
  for char in header['STARTIME']:
      if char == 'T':
          break  # Stop once 'T' is found.
      start_time += char  # Collect characters before 'T'.
  return f"{start_time} {hr}:{min}:{sec}"

def plot_spectra(fits_data, window_size = 10, xlim = None, ylim = None, filtering = None):
  hdul = fits.open(fits_data)
  # record_array = hdul[1].data.view(np.recarray)
  x1, y1 = get_x_y(fits_data)

  if filtering == 'savgol':
    y_smooth = savgol_filter(y1, window_length=11, polyorder=3)
  # if filtering == 'lowess':
  #   y_smooth =

  if xlim:
    plt.xlim(0, xlim)
  if ylim:
    plt.ylim(0.0001, ylim)
  plt.semilogy(x1, y1, label='y1', linewidth = 0.5)
  if filtering:
    plt.semilogy(x1, y_smooth, label = "rolling mean")
  plt.title(fits_data)
  plt.show()
  hdul.close()

def get_lat_long(xml_data):
  lat_long_path = "{https://isda.issdc.gov.in/pds4/isda/v1}"
  tree = ET.parse(xml_data)
  root = tree.getroot()
  latitude = []
  longitude = []
  for z in ["latitude","longitude"]:
    for x in ["lower_left", "upper_left", "upper_right", "lower_right"]:
      for d in root.iter():
        if d.tag == lat_long_path+f"{x}_{z}":
          if z == "latitude":
            latitude += [float(d.text)]
          else:
            longitude += [float(d.text)]
  return latitude, longitude

# Define the latitude and longitude of the 4 corners of the rectangle
def plot_patches(latitudes, longitudes, fig, update_flag = False):
  corners = {
      "lon": longitudes,  # Longitudes of the 4 corners
      "lat": latitudes,    # Latitudes of the 4 corners
  }

  # Add the first point again to close the loop of the polygon
  corners["lon"].append(corners["lon"][0])
  corners["lat"].append(corners["lat"][0])

  mean_lat = np.mean(corners["lat"])
  mean_lon = np.mean(corners["lon"])


  # Plot the rectangular patch as a polygon
  # if update_flag:
  fig.add_trace(go.Scattergeo(
      lon=corners["lon"],
      lat=corners["lat"],
      mode='lines+markers',
      fill='toself',  # Fill the polygon with color
      line=dict(width=2, color='red'),
      marker=dict(size=5, color='black'),
      name='Lunar Region'
  ))

  # Configure the layout to use an orthographic projection (Moon-like appearance)
  if update_flag:
    fig.update_geos(
        projection_type="orthographic",
        showland=True,        # Show land for visualization
        landcolor="lightgray",
        showocean=True,
        oceancolor="black",
        lonaxis=dict(showgrid=True, gridcolor="white", range=[mean_lon - 1, mean_lon + 1]),
        lataxis=dict(showgrid=True, gridcolor="white", range=[mean_lat - 1, mean_lat + 1])
    )

  # Update the layout for better aesthetics
  fig.update_layout(
      title='Patches on the Moon',
      height=700,  # Adjust height for better view
      showlegend=True,
      template='plotly_dark'  # Optional: Use dark theme for aesthetics
  )

def get_xml_files(directory):
  xml_files = []
  for root, _, files in tqdm(os.walk(directory)):
    for file in files:
      if file.endswith(".xml") or '.xml?' in file:
        xml_files.append(os.path.join(root, file))
  return xml_files

def get_fits_files(directory):
  fits_files = []
  for root, _, files in tqdm(os.walk(directory)):
    for file in files:
      if file.endswith(".fits") or '.fits?' in file:
        fits_files.append(os.path.join(root, file))
  return fits_files

def lorentzian(x, amplitude, center, width = 0.000001):
  return amplitude * width / ((x - center)**2 + width**2)
  # return amplitude * np.exp(-(x - center)**2 / width ** 2)/(width * np.sqrt(2 * np.pi))

def custom_error(params, x, y):
  model = np.zeros_like(x)
  for i in range(0, len(params), 3):
    amplitude, center, width = params[i: i + 3]
    model += lorentzian(x, amplitude, center, width)
  # return np.sum(abs(y - model))
  return chi2(model, y)

def multi_lorentzian(x, *params):
  result = np.zeros_like(x)
  for i in range(0, len(params), 3):
    try:
      amplitude, center, width = params[i:i+3]
      result += lorentzian(x, amplitude, center, width)
      # result[int((center - width / 2) / 0.0135): int((center + width / 2) / 0.0135)] += lorentzian(x, amplitude, center, width)[int((center - width / 2 + 1) / 0.0135): int((center + width / 2 + 1) / 0.0135)]
    except:
      # print(f"{i/3}th Lorentzian")
      pass
  return result

def fit_multi_lorentzian(list_, lower_limit = 0, upper_limit = 27.648, elements = None, plot_semilog = True, wid_init = 0.01, plot = True, return_lorentzian = False):
  conversion_factor = 0.0135

  # Example data (replace with your actual data)
  if len(list_) == 1:
    fits_data = list_[0]
    with fits.open(fits_data) as hdul:
        record_array = hdul[1].data.view(np.recarray)
        x = record_array.CHANNEL
        y = record_array.COUNTS
        x1 = x * conversion_factor
  else:
    x1, y = list_
  x = x1[int(lower_limit/conversion_factor): int(upper_limit/conversion_factor)]
  y = y[int(lower_limit/conversion_factor): int(upper_limit/conversion_factor)]
  y = np.where(y == 0, 0.00001, y)

  amp_init = 1

  element_peak = {}
  element_peak['O'] = 0.5
  element_peak['Mg'] = 1.25
  element_peak['Al'] = 1.49
  element_peak['Si'] = 1.74
  element_peak['S'] = 2.1

  # Initial guesses for parameters (adjust based on your data)
  initial_guess = []  # Example: 2 Lorentzians
  for element in elements:
    initial_guess.append(amp_init)
    initial_guess.append(element_peak[element])
    initial_guess.append(wid_init)
  try:
    # Perform the fit
    popt, pcov = curve_fit(multi_lorentzian, x, y, p0=initial_guess)

    # Print optimized parameters
    print("Optimized Parameters:", popt)

    # Generate the fitted curve
    y_fit = multi_lorentzian(x, *popt)

    if plot:
      plt.ylim(0.001, 1000)
      if plot_semilog:
        plt.semilogy(x, y, label="Data")
        plt.semilogy(x, y_fit, label="Multi-Lorentzian Fit")
      else:
        plt.semilogy(x, y, label="Data")
        plt.plot(x, y_fit, label="Multi-Lorentzian Fit")

      for i in range(0, len(popt), 3):
        plt.semilogy(x, lorentzian(x, popt[i], popt[i+1], popt[i+2]), linestyle = '--')

      # plt.title(local_solar_timestamp(fits_data))
      plt.legend()
      plt.show()
    # print(chi2_contingency(y, y_fit)[0])
    if return_lorentzian:
      return y_fit, [[lorentzian(x, popt[i], popt[i+1], popt[i+2]), popt[i+1], popt[i+2]] for i in range(0, len(initial_guess), 3)]
    return y_fit

  except RuntimeError:
    if plot:
      plt.ylim(0.001, 1000)
      plt.semilogy(x, y, label="Data")
      # plt.title(local_solar_timestamp(fits_data))
      plt.legend()
      plt.show()
    print("Optimal parameters not found. Try adjusting initial guesses or check data.")
    return np.zeros_like(x)

def trapz_method(lordata, mean, width):
  lower_bound = mean - width
  upper_bound = mean + width
  relevant = lordata[(int)(lower_bound/0.0135): (int)(upper_bound/0.0135)]
  # print(relevant)
  return np.trapz(relevant, [i for i in range(len(relevant))])

def earth_mover(x = None, y1 = None, y2 = None, file_path = True, norm = False):
  left = 1
  right = 5

  y1_ = y1[np.where((x < right) & (x > left))]
  y2_ = y2[np.where((x < right) & (x > left))]

  if norm:
    # Log transformation with condition
    y1_ = np.where(y1_ > 1, np.log(y1_), 0)
    y2_ = np.where(y2_ > 1, np.log(y2_), 0)

    y1_ = savgol_filter(y1_, window_length=11, polyorder=3)
    y2_ = savgol_filter(y2_, window_length=11, polyorder=3)

    # Convert to pandas Series for rolling mean
    y1_series = pd.Series(y1_)
    y2_series = pd.Series(y2_)

    # Apply a rolling mean with a specified window size (e.g., window=5)
    # Adjust the window size as needed
    y1_rolling = y1_series.rolling(window=10, min_periods=1).mean()
    y2_rolling = y2_series.rolling(window=10, min_periods=1).mean()

    # Convert rolling results back to numpy arrays
    y1_rolling = y1_rolling.to_numpy()
    y2_rolling = y2_rolling.to_numpy()

    # Proceed with min-max scaling
    y1_ = (y1_rolling - np.min(y1_rolling)) / (np.max(y1_rolling) - np.min(y1_rolling))
    y2_ = (y2_rolling - np.min(y2_rolling)) / (np.max(y2_rolling) - np.min(y2_rolling))

  emd = wasserstein_distance(y1_, y2_)

  # print("Earth Mover's Distance:", emd)
  return emd