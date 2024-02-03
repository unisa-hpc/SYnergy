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
df = pd.DataFrame()

df['n_kernels'] = data['n_kernels']
df['app_device_time'] = data['app_device_time_Median'] - data['app_freq_change_overhead_Max']
df['app_overhead_time'] = data['app_freq_change_overhead_Max']
df['phase_device_time'] = data['phase_device_time_Median'] - data['phase_freq_change_overhead_Max']
df['phase_overhead_time'] = data['phase_freq_change_overhead_Max']
df['kernel_device_time'] = data['kernel_device_time_Median'] - data['kernel_freq_change_overhead_Max']
df['kernel_overhead_time'] = data['kernel_freq_change_overhead_Max']

KERNEL_HATCHES = '/////'
PHASE_HATCHES = '------'

bar_width = 0.4
x = np.arange(len(df['n_kernels'])) * 1.5
plt.bar(x - bar_width, df['app_device_time'], width=bar_width, color='blue')
plt.bar(x - bar_width, df['app_overhead_time'], width=bar_width, bottom=df['app_device_time'], color='orange')

plt.bar(x, df['phase_device_time'], width=bar_width, color='blue', hatch=PHASE_HATCHES)
plt.bar(x, df['phase_overhead_time'], width=bar_width, bottom=df['phase_device_time'],  hatch=PHASE_HATCHES, color='orange')

plt.bar(x + bar_width, df['kernel_device_time'], width=bar_width, color='blue', hatch=KERNEL_HATCHES)
plt.bar(x + bar_width, df['kernel_overhead_time'], width=bar_width, bottom=df['kernel_device_time'],  hatch=KERNEL_HATCHES, color='orange')


legend_colors = [Patch(facecolor='blue', label='Device'), Patch(facecolor='orange', label='Host')]
legend_hatches = [
                  Patch(facecolor='blue', edgecolor='w', hatch=KERNEL_HATCHES, label=''), 
                  Patch(facecolor='blue', edgecolor='w', hatch=PHASE_HATCHES, label=''), 
                  Patch(facecolor='blue', label=''),
                  Patch(facecolor='orange', edgecolor='w', hatch=KERNEL_HATCHES, label='Kernel'), 
                  Patch(facecolor='orange', edgecolor='w', hatch=PHASE_HATCHES, label='Phase'), 
                  Patch(facecolor='orange', label='Application'),
                ]

# legend1 = plt.legend(handles=legend_colors, title='Energy Consumption', loc='center right', bbox_to_anchor=(0.8, 0.8))
# plt.gca().add_artist(legend1)
# legend2 = plt.legend(handles=legend_hatches, ncol=2,handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, title='Frequency Setting', loc='upper right')



plt.title('Title')
plt.xticks(x, df['n_kernels'], rotation=90)
plt.xlabel('# Kernels')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.savefig('plots.pdf')