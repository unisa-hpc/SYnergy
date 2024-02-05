import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from matplotlib.patches import Patch

if len(sys.argv) != 2:
  print('Usage: python plot_time.py <file_path>')
  exit(1)
file_path = sys.argv[1]

sns.set_theme()

data = pd.read_csv(file_path)
KERNEL_HATCHES = '///'
PHASE_HATCHES = '..'

bar_width = 0.3
x = np.arange(len(data['n_kernels']))
x_labels = [f'{x + x}' for x in data['n_kernels']]

# plotting time
df = pd.DataFrame()
df['n_kernels'] = data['n_kernels']
df['app_total_time'] = data['app_total_time_Average'] - data['app_freq_change_time_overhead_Average']
df['app_total_time_err'] = data['app_kernel_time_Stdev']
df['app_overhead_time'] = data['app_freq_change_time_overhead_Average']
df['app_overhead_time_err'] = data['app_freq_change_time_overhead_Stdev']

df['phase_total_time'] = data['phase_total_time_Average'] - data['phase_freq_change_time_overhead_Average']
df['phase_total_time_err'] = data['phase_kernel_time_Stdev'] 
df['phase_overhead_time'] = data['phase_freq_change_time_overhead_Average']
df['phase_overhead_time_err'] = data['phase_freq_change_time_overhead_Stdev']

df['kernel_total_time'] = data['kernel_total_time_Average'] - data['kernel_freq_change_time_overhead_Average']
df['kernel_total_time_err'] = data['kernel_kernel_time_Stdev'] 
df['kernel_overhead_time'] = data['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time_err'] = data['kernel_freq_change_time_overhead_Stdev']

plt.bar(x - bar_width, df['app_total_time'], width=bar_width, color='royalblue', yerr=df['app_total_time_err'])
plt.bar(x - bar_width, df['app_overhead_time'], width=bar_width, bottom=df['app_total_time'], color='darkorange', yerr=df['app_overhead_time_err'])

plt.bar(x, df['phase_total_time'], width=bar_width, color='royalblue', hatch=PHASE_HATCHES, yerr=df['phase_total_time_err'])
plt.bar(x, df['phase_overhead_time'], width=bar_width, bottom=df['phase_total_time'],  hatch=PHASE_HATCHES, color='darkorange', yerr=df['phase_overhead_time_err'])

plt.bar(x + bar_width, df['kernel_total_time'], width=bar_width, color='royalblue', hatch=KERNEL_HATCHES, yerr=df['kernel_total_time_err'])
plt.bar(x + bar_width, df['kernel_overhead_time'], width=bar_width, bottom=df['kernel_total_time'],  hatch=KERNEL_HATCHES, color='darkorange', yerr=df['kernel_overhead_time_err'])


legend = [Patch(facecolor='royalblue', label='Computation Time'), 
          Patch(facecolor='darkorange', label='Frequency Change Time'),
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
df['n_kernels'] = data['n_kernels']
df['app_energy'] = data['app_device_energy_Average']
df['app_energy_err'] = data['app_device_energy_Stdev']
df['phase_energy'] = data['phase_device_energy_Average']
df['phase_energy_err'] = data['phase_device_energy_Stdev']
df['kernel_energy'] = data['kernel_device_energy_Average']
df['kernel_energy_err'] = data['kernel_device_energy_Stdev']

plt.bar(x - bar_width, df['app_energy'], width=bar_width, label='Per-App Frequency Change', yerr=df['app_energy_err'])
plt.bar(x, df['phase_energy'], width=bar_width, label='Per-Phase Frequency Change', yerr=df['phase_energy_err'])
plt.bar(x + bar_width, df['kernel_energy'], width=bar_width, label='Per-Kernel Frequency Change', yerr=df['kernel_energy_err'])
plt.legend()

plt.xticks(x, x_labels)
plt.xlabel('Num. Kernel Calls')
plt.ylabel('Energy (J)')

plt.tight_layout()
plt.savefig('energy_overhead.pdf', bbox_inches="tight")
