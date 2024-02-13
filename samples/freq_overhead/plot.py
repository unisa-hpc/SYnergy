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

sns.set_theme()

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
KERNEL_HATCHES = '///'
PHASE_HATCHES = '..'

bar_width1 = 0.3
bar_width2 = 0.3
x = np.arange(len(data1['n_kernels']))
x_labels = [f'{x + x}' for x in data1['n_kernels']]

# plotting time
df = pd.DataFrame()
df['n_kernels'] = data1['n_kernels']
df['app_kernel_time'] = data1['app_kernel_time_Average']
df['app_kernel_time_err'] = data1['app_kernel_time_Stdev']
df['app_overhead_time1'] = data1['app_freq_change_time_overhead_Average']
df['app_overhead_time_err1'] = data1['app_freq_change_time_overhead_Stdev']

df['phase_kernel_time'] = data1['phase_kernel_time_Average']
df['phase_kernel_time_err'] = data1['phase_kernel_time_Stdev'] 
df['phase_overhead_time1'] = data1['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err1'] = data1['phase_freq_change_time_overhead_Stdev']

df['kernel_kernel_time'] = data1['kernel_kernel_time_Average']
df['kernel_kernel_time_err'] = data1['kernel_kernel_time_Stdev'] 
df['kernel_overhead_time1'] = data1['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err1'] = data1['kernel_freq_change_time_overhead_Stdev']

df['app_overhead_time2'] = data2['app_freq_change_time_overhead_Average']
df['app_overhead_time_err2'] = data2['app_freq_change_time_overhead_Stdev']

df['phase_overhead_time2'] = data2['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err2'] = data2['phase_freq_change_time_overhead_Stdev']

df['kernel_overhead_time2'] = data2['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err2'] = data2['kernel_freq_change_time_overhead_Stdev']

plt.bar(x - bar_width1, df['app_kernel_time'], width=bar_width1, color='royalblue', yerr=df['app_kernel_time_err'])
plt.bar(x - bar_width1, df['app_overhead_time1'], width=bar_width1, bottom=df['app_kernel_time'], color='darkorange', yerr=df['app_overhead_time_err1'])
plt.bar(x - bar_width1, df['app_overhead_time2'], width=bar_width1, bottom=df['app_kernel_time'], alpha=0.5, color='peru', yerr=df['app_overhead_time_err2'])

plt.bar(x, df['phase_kernel_time'], width=bar_width1, color='royalblue', hatch=PHASE_HATCHES, yerr=df['phase_kernel_time_err'])
plt.bar(x, df['phase_overhead_time1'], width=bar_width1, bottom=df['phase_kernel_time'], hatch=PHASE_HATCHES, color='darkorange', yerr=df['phase_overhead_time_err1'])
plt.bar(x, df['phase_overhead_time2'], width=bar_width1, bottom=df['phase_kernel_time'], alpha=0.5, hatch=PHASE_HATCHES, color='peru', yerr=df['phase_overhead_time_err2'])

plt.bar(x + bar_width1, df['kernel_kernel_time'], width=bar_width1, color='royalblue', hatch=KERNEL_HATCHES, yerr=df['kernel_kernel_time_err'])
plt.bar(x + bar_width1, df['kernel_overhead_time1'], width=bar_width1, bottom=df['kernel_kernel_time'], hatch=KERNEL_HATCHES, color='darkorange', yerr=df['kernel_overhead_time_err1'])
plt.bar(x + bar_width1, df['kernel_overhead_time2'], width=bar_width1, bottom=df['kernel_kernel_time'], alpha=0.5, hatch=KERNEL_HATCHES, color='peru', yerr=df['kernel_overhead_time_err2'])

legend = [Patch(facecolor='royalblue', label='Computation Time'), 
          Patch(facecolor='darkorange', label='Native Frequency Change Time'),
          Patch(facecolor='peru', alpha=0.5, label='GEOPM Frequency Change Time'),
          Patch(facecolor='none', edgecolor='k', label='Per-App Frequency Change'),
          Patch(facecolor='none', edgecolor='k', hatch=PHASE_HATCHES + ".", label='Per-Phase Frequency Change'),
          Patch(facecolor='none', edgecolor='k', hatch=KERNEL_HATCHES, label='Per-Kernel Frequency Change'),
          ]

legend1 = plt.legend(handles=legend)

plt.xticks(x, x_labels)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.savefig('time_overhead.pdf', bbox_inches="tight")

plt.clf()

# plotting energy
df = pd.DataFrame()
df['n_kernels'] = data1['n_kernels']
df['app_energy1'] = data1['app_kernel_energy_Average']
df['app_energy_err1'] = data1['app_kernel_energy_Stdev']
df['phase_energy1'] = data1['phase_kernel_energy_Average']
df['phase_energy_err1'] = data1['phase_kernel_energy_Stdev']
df['kernel_energy1'] = data1['kernel_kernel_energy_Average']
df['kernel_energy_err1'] = data1['kernel_kernel_energy_Stdev']
df['app_energy2'] = data2['app_device_energy_Average'] - data1['app_device_energy_Average']
df['app_energy_err2'] = data2['app_device_energy_Stdev']
df['phase_energy2'] = data2['phase_device_energy_Average'] - data1['phase_device_energy_Average']
df['phase_energy_err2'] = data2['phase_device_energy_Stdev']
df['kernel_energy2'] = data2['kernel_device_energy_Average'] - data1['kernel_device_energy_Average']
df['kernel_energy_err2'] = data2['kernel_device_energy_Stdev']

COLOR1 = 'C0'
COLOR2 = 'C1'
COLOR3 = 'C2'

plt.bar(x - bar_width2, df['app_energy1'], width=bar_width2, label='Per-App Frequency Change', yerr=df['app_energy_err1'], color=COLOR1)
plt.bar(x, df['phase_energy1'], width=bar_width2, label='Per-Phase Frequency Change', yerr=df['phase_energy_err1'], color=COLOR2)
plt.bar(x + bar_width2, df['kernel_energy1'], width=bar_width2, label='Per-Kernel Frequency Change', yerr=df['kernel_energy_err1'], color=COLOR3)

plt.bar(x - bar_width2, df['app_energy2'], bottom=df['app_energy1'], width=bar_width2, label='Per-App Frequency Change', yerr=df['app_energy_err2'], color=COLOR1, hatch=KERNEL_HATCHES, alpha=0.6)
plt.bar(x, df['phase_energy2'], bottom=df['phase_energy1'], width=bar_width2, label='Per-Phase Frequency Change', yerr=df['phase_energy_err2'], color=COLOR2, hatch=KERNEL_HATCHES, alpha=0.6)
plt.bar(x + bar_width2, df['kernel_energy2'], bottom=df['kernel_energy1'], width=bar_width2, label='Per-Kernel Frequency Change', yerr=df['kernel_energy_err2'], color=COLOR3, hatch=KERNEL_HATCHES, alpha=0.6)
plt.legend()

legend = [Patch(facecolor=COLOR1, label='Per-App Frequency Change'), 
          Patch(facecolor=COLOR2, label='Per-Phase Frequency Change'),
          Patch(facecolor=COLOR3, label='Per-Kernel Frequency Change'),
          Patch(facecolor='none', edgecolor='k', label='NATIVE Approach'),
          Patch(facecolor='none', edgecolor='k', hatch=KERNEL_HATCHES, label='GEOPM Approach'),
          ]

legend1 = plt.legend(handles=legend)

plt.xticks(x, x_labels)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Energy (J)')

plt.tight_layout()
plt.savefig('energy_overhead.pdf', bbox_inches="tight")
