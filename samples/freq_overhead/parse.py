import re
import pandas as pd
import sys

def parse_log_file_refined(file_path):
    # Define regex patterns for extracting data
    freq_pattern = re.compile(r'\[\*\] Running benchmark for frequency (\d+)')
    type_pattern = re.compile(r'(App|Kernel|Phase) frequency setting\.\.\.')
    energy_sample_before_pattern = re.compile(r'energy-sample-before\[J\]: ([\d.]+)')
    energy_sample_after_pattern = re.compile(r'energy-sample-after\[J\]: ([\d.]+)')
    energy_sample_delta_pattern = re.compile(r'energy-sample-delta\[J\]: ([\d.]+)')
    energy_sample_time_pattern = re.compile(r'energy-sample-time\[ms\]: ([\d.]+)')
    time_value_pattern = re.compile(r'device-time\[ms\]: \[ (.+) \]')
    device_energy_value_pattern = re.compile(r'device-energy\[J\]: \[ (.+) \]')
    host_energy_value_pattern = re.compile(r'host-energy\[J\]: \[ (.+) \]')
    avg_pattern = re.compile(r'(.+)-avg\[(ms|J)\]: ([\d.]+)')
    stdev_pattern = re.compile(r'(.+)-stdev\[(ms|J)\]: ([\d.]+)')
    max_pattern = re.compile(r'(.+)-max\[(ms|J)\]: ([\d.]+)')
    min_pattern = re.compile(r'(.+)-min\[(ms|J)\]: ([\d.]+)')
    median_pattern = re.compile(r'(.+)-median\[(ms|J)\]: ([\d.]+)')

    # Initialize variables to store the parsed data
    data = {}
    current_freq = None

    with open(file_path, 'r') as file:
        for line in file:
            # Check for frequency
            freq_match = freq_pattern.match(line)
            if freq_match:
                current_freq = int(freq_match.group(1))
                if current_freq not in data:
                    data[current_freq] = {'freq': current_freq}
                continue

            # Check for frequency type
            type_match = type_pattern.match(line)
            if type_match:
                current_type = type_match.group(1).lower()  # 'App', 'Kernel' or 'Phase'
                continue               

            # Extract and store data
            if current_freq is not None:
                prefix = f'{current_type}_'
                
                energy_sample_before_match = energy_sample_before_pattern.match(line)
                if energy_sample_before_match:
                    energy_sample_before = energy_sample_before_match.group(1)
                    data[current_freq][f'{prefix}energy_sample_before'] = float(energy_sample_before)
                    continue
                
                energy_sample_after_match = energy_sample_after_pattern.match(line)
                if energy_sample_after_match:
                    energy_sample_after = energy_sample_after_match.group(1)
                    data[current_freq][f'{prefix}energy_sample_after'] = float(energy_sample_after)
                    continue
                
                energy_sample_delta_match = energy_sample_delta_pattern.match(line)
                if energy_sample_delta_match:
                    energy_sample_delta = energy_sample_delta_match.group(1)
                    data[current_freq][f'{prefix}energy_sample_delta'] = float(energy_sample_delta)
                    continue
                
                energy_sample_time_match = energy_sample_time_pattern.match(line)
                if energy_sample_time_match:
                    energy_sample_time = energy_sample_time_match.group(1)
                    data[current_freq][f'{prefix}energy_sample_time'] = float(energy_sample_time)
                    continue 
                
                time_value_match = time_value_pattern.match(line)
                device_energy_value_match = device_energy_value_pattern.match(line)
                host_energy_value_match = host_energy_value_pattern.match(line)

                # For device-time, device-energy, and host-energy
                for match, metric in zip([time_value_match, device_energy_value_match, host_energy_value_match], 
                                         ['device_time', 'device_energy', 'host_energy']):
                    if match:
                        # Extract other statistics
                        for _ in range(5):
                            stat_line = next(file)
                            avg_match = avg_pattern.match(stat_line)
                            if avg_match:
                                data[current_freq][f'{prefix}{metric}_Average'] = float(avg_match.group(3))
                            stdev_match = stdev_pattern.match(stat_line)
                            if stdev_match:
                                data[current_freq][f'{prefix}{metric}_Stdev'] = float(stdev_match.group(3))
                            max_match = max_pattern.match(stat_line)
                            if max_match:
                                data[current_freq][f'{prefix}{metric}_Max'] = float(max_match.group(3))
                            min_match = min_pattern.match(stat_line)
                            if min_match:
                                data[current_freq][f'{prefix}{metric}_Min'] = float(min_match.group(3))
                            median_match = median_pattern.match(stat_line)
                            if median_match:
                                data[current_freq][f'{prefix}{metric}_Median'] = float(median_match.group(3))
    
    return pd.DataFrame.from_dict(data, orient='index')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python parse.py <input_file_path> <output_file_path>')
        exit(1)
    # Get the path to the log file
    file_path = sys.argv[1]
    output_path = sys.argv[2]
    # Parse the log file with the refined approach
    refined_dataset = parse_log_file_refined(file_path)
    refined_dataset.to_csv(output_path, index=False)