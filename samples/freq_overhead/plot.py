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
sns.lineplot(x='freq', y='nonsetting_device_time_Median', data=data, label='Non-Setting')
sns.lineplot(x='freq', y='setting_device_time_Median', data=data, label='Setting')
plt.title('Device Time Median for Setting vs Non-Setting Frequencies')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('Frequency')
plt.ylabel('Median Device Time (ms)')
plt.legend()

plt.subplot(3, 1, 2)
sns.lineplot(x='freq', y='nonsetting_device_energy_Median', data=data, label='Non-Setting')
sns.lineplot(x='freq', y='setting_device_energy_Median', data=data, label='Setting')
plt.title('Device Energy Consumption Median for Setting vs Non-Setting Frequencies')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('Frequency')
plt.ylabel('Median Device Energy Consumption (J)')
plt.legend()

plt.subplot(3, 1, 3)
sns.lineplot(x='freq', y='nonsetting_host_energy_Median', data=data, label='Non-Setting')
sns.lineplot(x='freq', y='setting_host_energy_Median', data=data, label='Setting')
plt.title('Host Energy Consumption Median for Setting vs Non-Setting Frequencies')
plt.xticks(data['freq'], rotation=90)
plt.xlabel('Frequency')
plt.ylabel('Average Host Energy Consumption (J)')
plt.legend()

plt.tight_layout()
plt.savefig('plots.pdf')