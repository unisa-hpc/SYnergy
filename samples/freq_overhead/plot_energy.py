import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from matplotlib.patches import Patch


fontdict_val={'fontsize': 8, 'style': 'italic'}
GEOPM_HATCHES = '///'
legend_props = {'size': 10}
text_y_pos = 0
rotation=20
padding=15
bar_width_offset = 0.01
bar_width = 0.15
COLOR1 = 'C0'
COLOR1 = 'C1'

if len(sys.argv) != 5:
  print('Usage: python plot_time.py <energy> <first> <second> <outfile>')
  exit(1)
  
energy_type = sys.argv[1]
file_path1 = sys.argv[2]
file_path2 = sys.argv[3]
outfile = sys.argv[4]

sns.set_theme()

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

max_energy = max(data1['app_device_energy_Average'].max(), data1['phase_device_energy_Average'].max(), data1['kernel_device_energy_Average'])
max_energy = max(max_energy, data1['app_host_energy_Average'].max(), data1['phase_host_energy_Average'].max(), data1['kernel_host_energy_Average'])
max_energy = max(max_energy, data2['app_device_energy_Average'].max(), data2['phase_device_energy_Average'].max(), data2['kernel_device_energy_Average'])
max_energy = max(max_energy, data2['app_host_energy_Average'].max(), data2['phase_host_energy_Average'].max(), data2['kernel_host_energy_Average'])
max_energy += max_energy % 50

x = np.arange(len(data1['n_kernels']))
x_labels = [f'{x + x}' for x in data1['n_kernels']]

# plotting energy
df = pd.DataFrame()
df['n_kernels'] = data1['n_kernels']
df['app_energy1'] = data1[f'app_{energy_type}_energy_Average'] #+ data1['app_host_energy_Average']
df['app_energy_err1'] = data1[f'app_{energy_type}_energy_Stdev'] #+ data1['app_host_energy_Stdev']
df['phase_energy1'] = data1[f'phase_{energy_type}_energy_Average'] #+ data1['phase_host_energy_Average']
df['phase_energy_err1'] = data1[f'phase_{energy_type}_energy_Stdev'] #+ data1['phase_host_energy_Stdev']
df['kernel_energy1'] = data1[f'kernel_{energy_type}_energy_Average'] #+ data1['kernel_host_energy_Average']
df['kernel_energy_err1'] = data1[f'kernel_{energy_type}_energy_Stdev'] #+ data1['kernel_host_energy_Stdev']
df['app_energy2'] = data2[f'app_{energy_type}_energy_Average']  #+ data2['app_host_energy_Average'] 
df['app_energy_err2'] = data2[f'app_{energy_type}_energy_Stdev'] #+ data2['app_host_energy_Stdev']
df['phase_energy2'] = data2[f'phase_{energy_type}_energy_Average']  #+ data2['phase_host_energy_Average'] 
df['phase_energy_err2'] = data2[f'phase_{energy_type}_energy_Stdev'] #+ data2['phase_host_energy_Stdev']
df['kernel_energy2'] = data2[f'kernel_{energy_type}_energy_Average'] #+ data2['kernel_host_energy_Average']
df['kernel_energy_err2'] = data2[f'kernel_{energy_type}_energy_Stdev'] #+ data2['kernel_host_energy_Stdev']

COLOR1 = 'C0'
COLOR2 = 'C1'

#device energy
plt.bar(x - bar_width - (3 * bar_width / 2) + bar_width_offset, df['app_energy1'], width=bar_width - bar_width_offset, yerr=df['app_energy_err1'], color=COLOR1)
plt.bar(x - (3 * bar_width / 2), df['app_energy2'], width=bar_width - bar_width_offset, yerr=df['app_energy_err2'], color=COLOR2)

plt.bar(x + bar_width - (3 * bar_width / 2) + bar_width_offset, df['phase_energy1'], width=bar_width - bar_width_offset, yerr=df['phase_energy_err1'], color=COLOR1)
plt.bar(x - bar_width + (3 * bar_width / 2), df['phase_energy2'], width=bar_width - bar_width_offset, yerr=df['phase_energy_err2'], color=COLOR2)

plt.bar(x + (3 * bar_width / 2) + bar_width_offset, df['kernel_energy1'], width=bar_width - bar_width_offset, yerr=df['kernel_energy_err1'], color=COLOR1)
plt.bar(x + bar_width + (3 * bar_width / 2), df['kernel_energy2'], width=bar_width - bar_width_offset, yerr=df['kernel_energy_err2'], color=COLOR2)

for xv in x:
  plt.text(xv - (bar_width / 2) - (bar_width * 3 / 2), text_y_pos, "Per-App", fontdict=fontdict_val, ha='center', va='top', rotation=rotation)
  plt.text(xv, text_y_pos, "Per-Phase", fontdict=fontdict_val, ha='center', va='top', rotation=rotation)
  plt.text(xv + (bar_width / 2) + (bar_width * 3 / 2), text_y_pos, "Per-Kernel", fontdict=fontdict_val, ha='center', va='top', rotation=rotation)


legend = [
          Patch(facecolor=COLOR1, label='NVML'), 
          Patch(facecolor=COLOR2, label='GEOPM'),
          ]

plt.legend(handles=legend, ncol=2, prop = legend_props)

plt.ylim(top=max_energy)
plt.xticks(x, x_labels)
for tick in plt.gca().xaxis.get_major_ticks():
  tick.set_pad(padding)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Energy (J)')
plt.savefig(outfile, bbox_inches="tight")
