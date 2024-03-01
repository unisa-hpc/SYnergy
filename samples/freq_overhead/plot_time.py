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
bar_width_offset = 0.01
bar_width = 0.13
padding=20
COLOR1 = 'C0'
COLOR2 = 'C1'

if len(sys.argv) != 4:
  print('Usage: python plot_time.py <first> <second> <outfile>')
  exit(1)
file_path1 = sys.argv[1]
file_path2 = sys.argv[2]
outfile = sys.argv[3]

sns.set_theme()

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
x = np.arange(len(data1['n_kernels']))
x_labels = [f'{x + x}' for x in data1['n_kernels']]

for key in data1.keys():
  if "time" in key:
    data1[key] = data1[key] / 1000
    data2[key] = data2[key] / 1000

# plotting time
df = pd.DataFrame()
df['n_kernels'] = data1['n_kernels']
df['app_time1'] = data1['app_kernel_time_Average']
df['app_time_err1'] = data1['app_kernel_time_Stdev']
df['app_overhead_time1'] = data1['app_freq_change_time_overhead_Average']
df['app_overhead_time_err1'] = data1['app_freq_change_time_overhead_Stdev']

df['phase_time1'] = data1['phase_kernel_time_Average']
df['phase_time_err1'] = data1['phase_kernel_time_Stdev'] 
df['phase_overhead_time1'] = data1['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err1'] = data1['phase_freq_change_time_overhead_Stdev']

df['kernel_time1'] = data1['kernel_kernel_time_Average']
df['kernel_time_err1'] = data1['kernel_kernel_time_Stdev'] 
df['kernel_overhead_time1'] = data1['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err1'] = data1['kernel_freq_change_time_overhead_Stdev']

df['app_time2'] = data2['app_kernel_time_Average']
df['app_time_err2'] = data2['app_kernel_time_Stdev']
df['app_overhead_time2'] = data2['app_freq_change_time_overhead_Average']
df['app_overhead_time_err2'] = data2['app_freq_change_time_overhead_Stdev']

df['phase_time2'] = data2['phase_kernel_time_Average']
df['phase_time_err2'] = data2['phase_kernel_time_Stdev'] 
df['phase_overhead_time2'] = data2['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err2'] = data2['phase_freq_change_time_overhead_Stdev']

df['kernel_time2'] = data1['kernel_kernel_time_Average']
df['kernel_time_err2'] = data1['kernel_kernel_time_Stdev'] 
df['kernel_overhead_time2'] = data2['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err2'] = data2['kernel_freq_change_time_overhead_Stdev']

plt.bar(x - bar_width - (bar_width * 3 / 2) + bar_width_offset, df['app_time1'], width=bar_width - bar_width_offset, color=COLOR1, yerr=df['app_time_err2'])
plt.bar(x - bar_width - (bar_width * 3 / 2) + bar_width_offset, df['app_overhead_time1'], width=bar_width - bar_width_offset, bottom=df['app_time1'], color=COLOR1, hatch=GEOPM_HATCHES, yerr=df['app_overhead_time_err1'])
# plt.bar(x , df['app_time2'], width=bar_width1, color=COLOR2_1, yerr=df['app_time_err2'])

plt.bar(x - (bar_width * 3 / 2), df['app_time2'], width=bar_width - bar_width_offset, color=COLOR2, yerr=df['app_time_err2'])
plt.bar(x - (bar_width * 3 / 2), df['app_overhead_time2'], width=bar_width - bar_width_offset, bottom=df['app_time2'], hatch=GEOPM_HATCHES, color=COLOR2, yerr=df['app_overhead_time_err2'])

plt.bar(x + bar_width - (bar_width * 3 / 2) + bar_width_offset, df['phase_time1'], width=bar_width - bar_width_offset, color=COLOR1, yerr=df['phase_time_err1'])
plt.bar(x + bar_width - (bar_width * 3 / 2) + bar_width_offset, df['phase_overhead_time1'], width=bar_width - bar_width_offset, bottom=df['phase_time1'], color=COLOR1, hatch=GEOPM_HATCHES, yerr=df['phase_overhead_time_err1'])

plt.bar(x - bar_width + (bar_width * 3 / 2), df['phase_time2'], width=bar_width - bar_width_offset, color=COLOR2, yerr=df['phase_time_err2'])
plt.bar(x - bar_width + (bar_width * 3 / 2), df['phase_overhead_time2'], width=bar_width - bar_width_offset, bottom=df['phase_time2'], hatch=GEOPM_HATCHES, color=COLOR2, yerr=df['phase_overhead_time_err2'])
# plt.bar(x + (bar_width1 * 3 / 2), df['phase_time2'], width=bar_width1, color=COLOR2_1, hatch=PHASE_HATCHES, yerr=df['phase_time_err2'])
# plt.bar(x + (bar_width1 * 3 / 2), df['phase_overhead_time2'], width=bar_width1, bottom=df['phase_time2'], hatch=PHASE_HATCHES, color=COLOR2_2, yerr=df['phase_overhead_time_err2'])

plt.bar(x + (bar_width * 3 / 2) + bar_width_offset, df['kernel_time1'], width=bar_width - bar_width_offset, color=COLOR1, yerr=df['kernel_time_err1'])
plt.bar(x + (bar_width * 3 / 2) + bar_width_offset, df['kernel_overhead_time1'], width=bar_width - bar_width_offset, bottom=df['kernel_time1'], color=COLOR1, hatch=GEOPM_HATCHES, yerr=df['kernel_overhead_time_err1'])

plt.bar(x + bar_width + (bar_width * 3 / 2), df['kernel_time2'], width=bar_width - bar_width_offset, color=COLOR2, yerr=df['kernel_time_err2'])
plt.bar(x + bar_width + (bar_width * 3 / 2), df['kernel_overhead_time2'], width=bar_width - bar_width_offset, bottom=df['kernel_time2'], hatch=GEOPM_HATCHES, color=COLOR2, yerr=df['kernel_overhead_time_err2'])

for xv in x:
  plt.text(xv - (bar_width / 2) - (bar_width * 3 / 2), text_y_pos, "Per-App", fontdict=fontdict_val, ha='center', va='top', rotation=rotation)
  plt.text(xv, text_y_pos, "Per-Phase", fontdict=fontdict_val, ha='center', va='top', rotation=rotation)
  plt.text(xv + (bar_width / 2) + (bar_width * 3 / 2), text_y_pos, "Per-Kernel", fontdict=fontdict_val, ha='center', va='top', rotation=rotation)


legend = [
          Patch(facecolor=COLOR1, label='NVML'), 
          Patch(facecolor=COLOR2, label='GEOPM'),
          # Patch(facecolor='none', edgecolor='k', hatch=PHASE_HATCHES + ".", label='Per-Phase'),
          Patch(facecolor='none', edgecolor='k', label='Computation Time'),
          Patch(facecolor='none', edgecolor='k', hatch=GEOPM_HATCHES, label='Freq. Change Overhead'),
          ]

legend1 = plt.legend(handles=legend, ncol=2, prop = legend_props)

plt.xticks(x, x_labels)
for tick in plt.gca().xaxis.get_major_ticks():
  tick.set_pad(padding)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Time (s)')

plt.tight_layout()
plt.savefig(outfile, bbox_inches="tight")

plt.clf()