import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

if len(sys.argv) != 2:
  print('Usage: python plot_time.py <file_path>')
  exit(1)
file_path = sys.argv[1]

data = pd.read_csv(file_path)

# discard each 3 items untill the number of elements is lower than 40
if len(data) > 40:
  data = data.iloc[::5, :]

plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
sns.lineplot(x='freq', y='nonsetting_device_time_Median', data=data, label='Pre-Setting Frequency')
sns.lineplot(x='freq', y='setting_device_time_Median', data=data, label='Runtime Frequency Setting')
plt.title('Device Time Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('Frequency')
plt.ylabel('Performances (ms)')
plt.legend()

plt.subplot(3, 1, 2)
sns.lineplot(x='freq', y='nonsetting_device_energy_Median', data=data, label='Pre-Setting Frequency')
sns.lineplot(x='freq', y='setting_device_energy_Median', data=data, label='Runtime Frequency Setting')
plt.title('Device Energy Consumption Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('Frequency')
plt.ylabel('Energy Consumption (J)')
plt.legend()

plt.subplot(3, 1, 3)
sns.lineplot(x='freq', y='nonsetting_host_energy_Median', data=data, label='Pre-Setting Frequency')
sns.lineplot(x='freq', y='setting_host_energy_Median', data=data, label='Runtime Frequency Setting')
plt.title('Host Energy Consumption Median')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('Frequency')
plt.ylabel('Energy Consumption (J)')
plt.legend()

plt.suptitle('Frequency Changing Overhead with 15 Kernels Execution')

plt.tight_layout()
plt.savefig('plots.pdf')