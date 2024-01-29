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

# discard each 3 items untill the number of elements is lower than 40
if len(data) > 40:
  data = data.iloc[::5, :]

plt.figure(figsize=(10, 14))
sns.set_theme()

plt.subplot(4, 1, 1)
sns.lineplot(x='freq', y='nonsetting_device_time_Median', data=data, label='Pre-Setting Frequency')
sns.lineplot(x='freq', y='setting_device_time_Median', data=data, label='Runtime Frequency Setting')
plt.title('Device Time Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Performances (ms)')
plt.legend()

plt.subplot(4, 1, 2)
sns.lineplot(x='freq', y='nonsetting_device_energy_Median', data=data, label='Pre-Setting Frequency')
sns.lineplot(x='freq', y='setting_device_energy_Median', data=data, label='Runtime Frequency Setting')
plt.title('Device Energy Consumption Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Energy Consumption (J)')
plt.legend()

plt.subplot(4, 1, 3)
sns.lineplot(x='freq', y='nonsetting_host_energy_Median', data=data, label='Pre-Setting Frequency')
sns.lineplot(x='freq', y='setting_host_energy_Median', data=data, label='Runtime Frequency Setting')
plt.title('Host Energy Consumption Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Energy Consumption (J)')
plt.legend()

plt.subplot(4, 1, 4)
# Take median for energy for both host and device
df = pd.DataFrame(data[['freq', 'nonsetting_device_energy_Median', 'nonsetting_host_energy_Median', 'setting_device_energy_Median', 'setting_host_energy_Median']])
# invert the order of the dataframe
df = df.iloc[::-1]

bar_width = 0.40
x = np.arange(len(df['freq']))
plt.bar(x - bar_width/2, df['nonsetting_device_energy_Median'], width=bar_width, color='blue')
plt.bar(x - bar_width/2, df['nonsetting_host_energy_Median'], width=bar_width, bottom=df['nonsetting_device_energy_Median'], color='orange')

# Stacking device_energy_B and host_energy_B
plt.bar(x + bar_width/2, df['setting_device_energy_Median'], width=bar_width, color='blue', hatch='/////')
plt.bar(x + bar_width/2, df['setting_host_energy_Median'], width=bar_width, bottom=df['setting_device_energy_Median'],  hatch='/////', color='orange')

legend_colors = [Patch(facecolor='blue', label='Device'), Patch(facecolor='orange', label='Host')]
legend_hatches = [
                  Patch(facecolor='blue', edgecolor='w', hatch='/////', label=''), 
                  Patch(facecolor='blue', label=''),
                  Patch(facecolor='orange', edgecolor='w', hatch='/////', label='Per-Kernel at Runtime'), 
                  Patch(facecolor='orange', label='Pre-Application Statically'),
                ]

legend1 = plt.legend(handles=legend_colors, title='Energy Consumption', loc='center right', bbox_to_anchor=(0.72, 0.8))
plt.gca().add_artist(legend1)
legend2 = plt.legend(handles=legend_hatches, ncol=2,handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, title='Frequency Setting', loc='upper right')



plt.title('Device and Host Energy Consumption Median')
plt.xticks(x, df['freq'], rotation=90)
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Energy Consumption (J)')

plt.suptitle('Frequency Changing Overhead with 15 Kernels Execution')

plt.tight_layout()
plt.savefig('plots.pdf')