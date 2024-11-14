import zipfile
from astropy.io import fits
import numpy as np
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from utils import *
import io

dir = '' # Enter your directory containing month wise zips here here.

comparison_file = '' #Enter noise file here
x_comp, y_comp = get_x_y(comparison_file)

#------------Enter the months of interest in the format (YYYY_MM)----------------#
zip_list = ['2021_05', '2021_06']
prefix = 'ch2_cla_l1_'
suffix = '.zip'

zips_list = []
for zips in zip_list:
  zips_list.append(prefix + zips + suffix)


# Saving parquets for each month with EarthMover Distance
for file in tqdm(zips_list):
  print(f"File: {file}")
  file_list = []
  distances = []
  distances_norm = []
  headers = []
  x_list = []
  y_list = []
  try:
    if file.endswith(".zip"):
      with zipfile.ZipFile(os.path.join(dir, file), 'r') as zip_ref:
        for file_name in tqdm(zip_ref.namelist()):
          if file_name.endswith('.fits'):
            with zip_ref.open(file_name) as filename:
              with fits.open(io.BytesIO(filename.read())) as hdul:
                header = hdul[1].header
                record_array = hdul[1].data.view(np.recarray)
                x, y = record_array.CHANNEL, record_array.COUNTS
                x = np.array(x, dtype = 'float64')
                x = x * 0.0135
                y = np.array(y, dtype = 'float64')
                hdul.close()
                distance = earth_mover((x_comp, y_comp), (x, y), file_path = False)
                distance_norm = earth_mover((x_comp, y_comp), (x, y), file_path = False, norm = True)

                file_list.append(file_name)
                headers.append(header)
                x_list.append(x)
                y_list.append(y)
                distances.append(distance)
                distances_norm.append(distance_norm)
      gc.collect()
      x_list_as_lists = [x.tolist() for x in x_list]
      y_list_as_lists = [y.tolist() for y in y_list]

      print("OK1")
      header_dict = {}
      for key in tqdm(headers[0]):
        header_dict[key] = []
        for header in headers:
          header_dict[key].append(header[key])

      print("OK1.5")

      dict_ = {'file': file_list, 'distance': distances, 'distance_norm': distances_norm, 'x': x_list_as_lists, 'y': y_list_as_lists}
      for key in tqdm(headers[0]):
        dict_[key] = header_dict[key]

      print("OK2")
      save = file.split('_')[3: ]
      save_str = f"{save[0]}_{((save[1].split('.'))[0])}"

      df = pd.DataFrame(dict_)

      table = pa.Table.from_pandas(df, preserve_index=False)

      # Write to Parquet
      pq.write_table(table, f'{dir}final_parquets/{save_str}.parquet')

      print("OK3")
  except:
    print(f"Error in {file}")