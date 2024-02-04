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

# plotting time
df = pd.DataFrame()
df['n_kernels'] = data['n_kernels']
df['app_total_time'] = data['app_total_time_Average'] - data['app_freq_change_time_overhead_Average']
df['app_overhead_time'] = data['app_freq_change_time_overhead_Average']
df['phase_total_time'] = data['phase_total_time_Average'] - data['phase_freq_change_time_overhead_Average']
df['phase_overhead_time'] = data['phase_freq_change_time_overhead_Average']
df['kernel_total_time'] = data['kernel_total_time_Average'] - data['kernel_freq_change_time_overhead_Average']
df['kernel_overhead_time'] = data['kernel_freq_change_time_overhead_Average']

plt.bar(x - bar_width, df['app_total_time'], width=bar_width, color='royalblue')
plt.bar(x - bar_width, df['app_overhead_time'], width=bar_width, bottom=df['app_total_time'], color='darkorange')

plt.bar(x, df['phase_total_time'], width=bar_width, color='royalblue', hatch=PHASE_HATCHES)
plt.bar(x, df['phase_overhead_time'], width=bar_width, bottom=df['phase_total_time'],  hatch=PHASE_HATCHES, color='darkorange')

plt.bar(x + bar_width, df['kernel_total_time'], width=bar_width, color='royalblue', hatch=KERNEL_HATCHES)
plt.bar(x + bar_width, df['kernel_overhead_time'], width=bar_width, bottom=df['kernel_total_time'],  hatch=KERNEL_HATCHES, color='darkorange')


legend = [Patch(facecolor='royalblue', label='Computation Time'), 
          Patch(facecolor='darkorange', label='Frequency Change Time'),
          Patch(facecolor='none', edgecolor='k', label='Per App Frequency Change'),
          Patch(facecolor='none', edgecolor='k', hatch=PHASE_HATCHES + ".", label='Per Phase Frequency Change'),
          Patch(facecolor='none', edgecolor='k', hatch=KERNEL_HATCHES, label='Per Kernel Frequency Change'),
          ]

legend1 = plt.legend(handles=legend)

plt.xticks(x, df['n_kernels'])
plt.xlabel('Number of Kernels')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.savefig('time_overhead.pdf', bbox_inches="tight")

plt.clf()

# plotting energy
df = pd.DataFrame()
df['n_kernels'] = data['n_kernels']
df['app_energy'] = data['app_device_energy_Average']
df['phase_energy'] = data['phase_device_energy_Average']
df['kernel_energy'] = data['kernel_device_energy_Average']

plt.bar(x - bar_width, df['app_energy'], width=bar_width, color='royalblue')
plt.bar(x, df['phase_energy'], width=bar_width, color='royalblue', hatch=PHASE_HATCHES)
plt.bar(x + bar_width, df['kernel_energy'], width=bar_width, color='royalblue', hatch=KERNEL_HATCHES)

legend = [Patch(facecolor='royalblue', edgecolor='k', label='Per App Frequency Change'),
          Patch(facecolor='royalblue', edgecolor='k', hatch=PHASE_HATCHES + ".", label='Per Phase Frequency Change'),
          Patch(facecolor='royalblue', edgecolor='k', hatch=KERNEL_HATCHES, label='Per Kernel Frequency Change'),
          ]

legend1 = plt.legend(handles=legend)

plt.xticks(x, df['n_kernels'])
plt.xlabel('Number of Kernels')
plt.ylabel('Energy (J)')

plt.tight_layout()
plt.savefig('energy_overhead.pdf', bbox_inches="tight")
