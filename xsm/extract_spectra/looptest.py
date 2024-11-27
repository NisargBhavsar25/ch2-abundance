import pandas as pd
import subprocess

df = pd.read_csv('subset.csv')
df1 = pd.read_csv("spectra_times.csv")

dates = df['date'].tolist()
start_times = df['STARTIME'].tolist()
stop_times = df['ENDTIME'].tolist()

for i in range(len(dates)):
    print(f'Processing {dates[i]}, {start_times[i]}, {stop_times[i]}')
    cmd = f'python solar_test.py . {dates[i]} ./test_spectrum ./test_spectrum/table {start_times[i]} {stop_times[i]}'
    subprocess.run(cmd, shell=True)
