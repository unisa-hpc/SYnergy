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

data = pd.read_csv(file_path)

KERNEL_HATCHES = '/////'
PHASE_HATCHES = '------'

# discard each 3 items untill the number of elements is lower than 40
if len(data) > 40:
  data = data.iloc[::5, :]

plt.figure(figsize=(10, 14))
sns.set_theme()

plt.subplot(4, 1, 1)
sns.lineplot(x='freq', y='app_device_time_Median', data=data, label='App Frequency Setting')
sns.lineplot(x='freq', y='phase_device_time_Median', data=data, label='Phase Frequency Setting')
sns.lineplot(x='freq', y='kernel_device_time_Median', data=data, label='Kernel Frequency Setting')
plt.title('Device Time Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Performances (ms)')
plt.legend()

plt.subplot(4, 1, 2)
sns.lineplot(x='freq', y='app_device_energy_Median', data=data, label='App Frequency Setting')
sns.lineplot(x='freq', y='phase_device_energy_Median', data=data, label='Phase Frequency Setting')
sns.lineplot(x='freq', y='kernel_device_energy_Median', data=data, label='Kernel Frequency Setting')
plt.title('Device Energy Consumption Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Energy Consumption (J)')
plt.legend()

plt.subplot(4, 1, 3)
sns.lineplot(x='freq', y='app_host_energy_Median', data=data, label='App Frequency Setting Setting')
sns.lineplot(x='freq', y='phase_host_energy_Median', data=data, label='Phase Frequency Setting')
sns.lineplot(x='freq', y='kernel_host_energy_Median', data=data, label='Kernel Frequency Setting')
plt.title('Host Energy Consumption Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Energy Consumption (J)')
plt.legend()

plt.subplot(4, 1, 4)
# invert the order of the dataframe
df = data.iloc[::-1]

bar_width = 0.4
x = np.arange(len(df['freq'])) * 1.5
plt.bar(x - bar_width, df['app_device_energy_Median'], width=bar_width, color='blue')
plt.bar(x - bar_width, df['app_host_energy_Median'], width=bar_width, bottom=df['app_device_energy_Median'], color='orange')

plt.bar(x, df['phase_device_energy_Median'], width=bar_width, color='blue', hatch=PHASE_HATCHES)
plt.bar(x, df['phase_host_energy_Median'], width=bar_width, bottom=df['phase_device_energy_Median'],  hatch=PHASE_HATCHES, color='orange')

plt.bar(x + bar_width, df['kernel_device_energy_Median'], width=bar_width, color='blue', hatch=KERNEL_HATCHES)
plt.bar(x + bar_width, df['kernel_host_energy_Median'], width=bar_width, bottom=df['kernel_device_energy_Median'],  hatch=KERNEL_HATCHES, color='orange')


legend_colors = [Patch(facecolor='blue', label='Device'), Patch(facecolor='orange', label='Host')]
legend_hatches = [
                  Patch(facecolor='blue', edgecolor='w', hatch=KERNEL_HATCHES, label=''), 
                  Patch(facecolor='blue', edgecolor='w', hatch=PHASE_HATCHES, label=''), 
                  Patch(facecolor='blue', label=''),
                  Patch(facecolor='orange', edgecolor='w', hatch=KERNEL_HATCHES, label='Kernel'), 
                  Patch(facecolor='orange', edgecolor='w', hatch=PHASE_HATCHES, label='Phase'), 
                  Patch(facecolor='orange', label='Application'),
                ]

legend1 = plt.legend(handles=legend_colors, title='Energy Consumption', loc='center right', bbox_to_anchor=(0.8, 0.8))
plt.gca().add_artist(legend1)
legend2 = plt.legend(handles=legend_hatches, ncol=2,handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, title='Frequency Setting', loc='upper right')



plt.title('Device and Host Energy Consumption Median')
plt.xticks(x, df['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Energy Consumption (J)')

plt.suptitle('Frequency Changing Overhead with 15 Kernels Execution')

plt.tight_layout()
plt.savefig('plots.pdf')