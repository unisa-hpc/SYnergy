import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from matplotlib.patches import Patch

if len(sys.argv) != 3:
  print('Usage: python plot_time.py <first> <second>')
  exit(1)
file_path1 = sys.argv[1]
file_path2 = sys.argv[2]

def lighten_color(color, amount=0.5):
  import matplotlib.colors as mc
  import colorsys
  try:
    c = mc.cnames[color]
  except:
    c = color
  c = colorsys.rgb_to_hls(*mc.to_rgb(c))
  return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

sns.set_theme()

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
KERNEL_HATCHES = '///'
PHASE_HATCHES = '..'

bar_width1 = 0.15
bar_width2 = 0.15
x = np.arange(len(data1['n_kernels']))
x_labels = [f'{x + x}' for x in data1['n_kernels']]

# plotting time
df = pd.DataFrame()
df['n_kernels'] = data1['n_kernels']
df['app_kernel_time1'] = data1['app_kernel_time_Average']
df['app_kernel_time_err1'] = data1['app_kernel_time_Stdev']
df['app_overhead_time1'] = data1['app_freq_change_time_overhead_Average']
df['app_overhead_time_err1'] = data1['app_freq_change_time_overhead_Stdev']

df['phase_kernel_time1'] = data1['phase_kernel_time_Average']
df['phase_kernel_time_err1'] = data1['phase_kernel_time_Stdev'] 
df['phase_overhead_time1'] = data1['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err1'] = data1['phase_freq_change_time_overhead_Stdev']

df['kernel_kernel_time1'] = data1['kernel_kernel_time_Average']
df['kernel_kernel_time_err1'] = data1['kernel_kernel_time_Stdev'] 
df['kernel_overhead_time1'] = data1['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err1'] = data1['kernel_freq_change_time_overhead_Stdev']

df['app_kernel_time2'] = data2['app_kernel_time_Average']
df['app_kernel_time_err2'] = data2['app_kernel_time_Stdev']
df['app_overhead_time2'] = data2['app_freq_change_time_overhead_Average']
df['app_overhead_time_err2'] = data2['app_freq_change_time_overhead_Stdev']

df['phase_kernel_time2'] = data2['phase_kernel_time_Average']
df['phase_kernel_time_err2'] = data2['phase_kernel_time_Stdev'] 
df['phase_overhead_time2'] = data2['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err2'] = data2['phase_freq_change_time_overhead_Stdev']

df['kernel_kernel_time2'] = data1['kernel_kernel_time_Average']
df['kernel_kernel_time_err2'] = data1['kernel_kernel_time_Stdev'] 
df['kernel_overhead_time2'] = data2['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err2'] = data2['kernel_freq_change_time_overhead_Stdev']

COLOR1_1 = 'royalblue'
COLOR1_2 = 'tab:orange'
COLOR2_1 = 'royalblue'
COLOR2_2 = 'tab:red'

plt.bar(x - bar_width1 - (bar_width1 * 3 / 2), df['app_kernel_time1'], width=bar_width1, color=COLOR1_1, yerr=df['app_kernel_time_err2'])
plt.bar(x - bar_width1 - (bar_width1 * 3 / 2), df['app_overhead_time1'], width=bar_width1, bottom=df['app_kernel_time1'], color=COLOR1_2, yerr=df['app_overhead_time_err1'])
# plt.bar(x , df['app_kernel_time2'], width=bar_width1, color=COLOR2_1, yerr=df['app_kernel_time_err2'])

plt.bar(x - (bar_width1 * 3 / 2), df['app_kernel_time2'], width=bar_width1, color=COLOR2_1, yerr=df['app_kernel_time_err2'])
plt.bar(x - (bar_width1 * 3 / 2), df['app_overhead_time2'], width=bar_width1, bottom=df['app_kernel_time2'], color=COLOR2_2, yerr=df['app_overhead_time_err2'])


plt.bar(x + bar_width1 - (bar_width1 * 3 / 2), df['phase_kernel_time1'], width=bar_width1, color=COLOR1_1, hatch=PHASE_HATCHES, yerr=df['phase_kernel_time_err1'])
plt.bar(x + bar_width1 - (bar_width1 * 3 / 2), df['phase_overhead_time1'], width=bar_width1, bottom=df['phase_kernel_time1'], hatch=PHASE_HATCHES, color=COLOR1_2, yerr=df['phase_overhead_time_err1'])

plt.bar(x - bar_width1 + (bar_width1 * 3 / 2), df['phase_kernel_time2'], width=bar_width1, color=COLOR2_1, hatch=PHASE_HATCHES, yerr=df['phase_kernel_time_err2'])
plt.bar(x - bar_width1 + (bar_width1 * 3 / 2), df['phase_overhead_time2'], width=bar_width1, bottom=df['phase_kernel_time2'], hatch=PHASE_HATCHES, color=COLOR2_2, yerr=df['phase_overhead_time_err2'])
# plt.bar(x + (bar_width1 * 3 / 2), df['phase_kernel_time2'], width=bar_width1, color=COLOR2_1, hatch=PHASE_HATCHES, yerr=df['phase_kernel_time_err2'])
# plt.bar(x + (bar_width1 * 3 / 2), df['phase_overhead_time2'], width=bar_width1, bottom=df['phase_kernel_time2'], hatch=PHASE_HATCHES, color=COLOR2_2, yerr=df['phase_overhead_time_err2'])

plt.bar(x + (bar_width1 * 3 / 2), df['kernel_kernel_time1'], width=bar_width1, color=COLOR1_1, hatch=KERNEL_HATCHES, yerr=df['kernel_kernel_time_err1'])
plt.bar(x + (bar_width1 * 3 / 2), df['kernel_overhead_time1'], width=bar_width1, bottom=df['kernel_kernel_time1'], hatch=KERNEL_HATCHES, color=COLOR1_2, yerr=df['kernel_overhead_time_err1'])

plt.bar(x + bar_width1 + (bar_width1 * 3 / 2), df['kernel_kernel_time2'], width=bar_width1, color=COLOR2_1, hatch=KERNEL_HATCHES, yerr=df['kernel_kernel_time_err2'])
plt.bar(x + bar_width1 + (bar_width1 * 3 / 2), df['kernel_overhead_time2'], width=bar_width1, bottom=df['kernel_kernel_time2'], hatch=KERNEL_HATCHES, color=COLOR2_2, yerr=df['kernel_overhead_time_err2'])

legend = [
          Patch(facecolor='none', edgecolor='k', label='Per-App Freq. Change'),
          Patch(facecolor='none', edgecolor='k', hatch=PHASE_HATCHES + ".", label='Per-Phase Freq. Change'),
          Patch(facecolor='none', edgecolor='k', hatch=KERNEL_HATCHES, label='Per-Kernel Freq. Change'),
          Patch(facecolor=COLOR1_1, label='Computation Time'), 
          Patch(facecolor=COLOR1_2, label='Native Overhead'),
          Patch(facecolor=COLOR2_2, label='GEOPM Overhead'),
          ]

legend1 = plt.legend(handles=legend, ncol=2)

plt.xticks(x, x_labels)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.savefig('TimeOverhead.pdf', bbox_inches="tight")

plt.clf()

# plotting energy
df = pd.DataFrame()
df['n_kernels'] = data1['n_kernels']
df['app_energy1'] = data1['app_device_energy_Average'] #+ data1['app_host_energy_Average']
df['app_energy_err1'] = data1['app_device_energy_Stdev'] #+ data1['app_host_energy_Stdev']
df['phase_energy1'] = data1['phase_device_energy_Average'] #+ data1['phase_host_energy_Average']
df['phase_energy_err1'] = data1['phase_device_energy_Stdev'] #+ data1['phase_host_energy_Stdev']
df['kernel_energy1'] = data1['kernel_device_energy_Average'] #+ data1['kernel_host_energy_Average']
df['kernel_energy_err1'] = data1['kernel_device_energy_Stdev'] #+ data1['kernel_host_energy_Stdev']
df['app_energy2'] = data2['app_device_energy_Average']  #+ data2['app_host_energy_Average'] 
df['app_energy_err2'] = data2['app_device_energy_Stdev'] #+ data2['app_host_energy_Stdev']
df['phase_energy2'] = data2['phase_device_energy_Average']  #+ data2['phase_host_energy_Average'] 
df['phase_energy_err2'] = data2['phase_device_energy_Stdev'] #+ data2['phase_host_energy_Stdev']
df['kernel_energy2'] = data2['kernel_device_energy_Average'] #+ data2['kernel_host_energy_Average']
df['kernel_energy_err2'] = data2['kernel_device_energy_Stdev'] #+ data2['kernel_host_energy_Stdev']

df['host_app_energy1'] = data1['app_host_energy_Average']
df['host_app_energy_err1'] = data1['app_host_energy_Stdev']
df['host_phase_energy1'] = data1['phase_host_energy_Average']
df['host_phase_energy_err1'] = data1['phase_host_energy_Stdev']
df['host_kernel_energy1'] = data1['kernel_host_energy_Average']
df['host_kernel_energy_err1'] = data1['kernel_host_energy_Stdev']
df['host_app_energy2'] = data2['app_host_energy_Average'] 
df['host_app_energy_err2'] = data2['app_host_energy_Stdev']
df['host_phase_energy2'] = data2['phase_host_energy_Average'] 
df['host_phase_energy_err2'] = data2['phase_host_energy_Stdev']
df['host_kernel_energy2'] = data2['kernel_host_energy_Average']
df['host_kernel_energy_err2'] = data2['kernel_host_energy_Stdev']

COLOR1_1 = 'C0'
COLOR2_1 = 'C1'
COLOR1 = lighten_color(COLOR1_1, 1.5)
COLOR2 = lighten_color(COLOR2_1, 1.5)
# COLOR3 = 'C2'
#device energy
plt.bar(x - bar_width2 - (3 * bar_width2 / 2), df['app_energy1'], width=bar_width2, yerr=df['app_energy_err1'], color=COLOR1)
plt.bar(x - (3 * bar_width2 / 2), df['app_energy2'], width=bar_width2, yerr=df['app_energy_err2'], color=COLOR2)

plt.bar(x + bar_width2 - (3 * bar_width2 / 2), df['phase_energy1'], width=bar_width2, yerr=df['phase_energy_err1'], color=COLOR1, hatch=PHASE_HATCHES)
plt.bar(x - bar_width2 + (3 * bar_width2 / 2), df['phase_energy2'], width=bar_width2, yerr=df['phase_energy_err2'], color=COLOR2, hatch=PHASE_HATCHES)

plt.bar(x + (3 * bar_width2 / 2), df['kernel_energy1'], width=bar_width2, yerr=df['kernel_energy_err1'], color=COLOR1, hatch=KERNEL_HATCHES)
plt.bar(x + bar_width2 + (3 * bar_width2 / 2), df['kernel_energy2'], width=bar_width2, yerr=df['kernel_energy_err2'], color=COLOR2, hatch=KERNEL_HATCHES)

# host energy
# plt.bar(x - bar_width2 - (3 * bar_width2 / 2), df['host_app_energy1'], bottom=df['app_energy1'], width=bar_width2, yerr=df['host_app_energy_err1'], color=COLOR1_1)
# plt.bar(x - (3 * bar_width2 / 2), df['host_app_energy2'], bottom=df['app_energy2'], width=bar_width2, yerr=df['host_app_energy_err2'], color=COLOR2_1)

# plt.bar(x + bar_width2 - (3 * bar_width2 / 2), df['host_phase_energy1'], bottom=df['phase_energy1'], width=bar_width2, yerr=df['host_phase_energy_err1'], color=COLOR1_1, hatch=PHASE_HATCHES)
# plt.bar(x - bar_width2 + (3 * bar_width2 / 2), df['host_phase_energy2'], bottom=df['phase_energy2'], width=bar_width2, yerr=df['host_phase_energy_err2'], color=COLOR2_1, hatch=PHASE_HATCHES)

# plt.bar(x + (3 * bar_width2 / 2), df['host_kernel_energy1'], bottom=df['kernel_energy1'], width=bar_width2, yerr=df['host_kernel_energy_err1'], color=COLOR1_1, hatch=KERNEL_HATCHES)
# plt.bar(x + bar_width2 + (3 * bar_width2 / 2), df['host_kernel_energy2'], bottom=df['kernel_energy2'], width=bar_width2, yerr=df['host_kernel_energy_err2'], color=COLOR2_1, hatch=KERNEL_HATCHES)

legend = [
          # Patch(facecolor=COLOR2_1, label='GEOPM Computation Time'),
          Patch(facecolor='none', edgecolor='k', label='Per-App Freq. Change'),
          Patch(facecolor='none', edgecolor='k', hatch=PHASE_HATCHES + ".", label='Per-Phase Freq. Change'),
          Patch(facecolor='none', edgecolor='k', hatch=KERNEL_HATCHES, label='Per-Kernel Freq. Change'),
          Patch(facecolor=COLOR1, label='Native'), 
          Patch(facecolor=COLOR2, label='GEOPM'),
          ]

legend1 = plt.legend(handles=legend, ncol=2)

plt.xticks(x, x_labels)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Energy (J)')

plt.tight_layout()
plt.savefig('EnergyOverhead.pdf', bbox_inches="tight")
