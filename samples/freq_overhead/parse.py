import re
import pandas as pd
import sys

def parse_log_file_refined(file_path):
    # Define regex patterns for extracting data
    nkernels_pattern = re.compile(r'Running freq_overhead for (\d+) iterations')
    type_pattern = re.compile(r'Policy: (app|kernel|phase)')
    energy_sample_before_pattern = re.compile(r'energy-sample-before\[J\]: ([\d.]+)')
    energy_sample_after_pattern = re.compile(r'energy-sample-after\[J\]: ([\d.]+)')
    energy_sample_delta_pattern = re.compile(r'energy-sample-delta\[J\]: ([\d.]+)')
    energy_sample_time_pattern = re.compile(r'energy-sample-time\[ms\]: ([\d.]+)')
    time_value_pattern = re.compile(r'total-time\[ms\]: \[ (.+) \]')
    kernel_value_pattern = re.compile(r'kernel-time\[ms\]: \[ (.+) \]')
    device_energy_value_pattern = re.compile(r'device-energy\[J\]: \[ (.+) \]')
    host_energy_value_pattern = re.compile(r'host-energy\[J\]: \[ (.+) \]')
    freq_change_time_overhead_pattern = re.compile(r'freq-change-time-overhead\[ms\]: \[ (.+) \]')
    freq_change_device_energy_overhead_pattern = re.compile(r'freq-change-device-energy-overhead\[J\]: \[ (.+) \]')
    freq_change_host_energy_overhead_pattern = re.compile(r'freq-change-host-energy-overhead\[J\]: \[ (.+) \]')
    avg_pattern = re.compile(r'(.+)-avg\[(ms|J)\]: ([\d.]+)')
    stdev_pattern = re.compile(r'(.+)-stdev\[(ms|J)\]: ([\d.]+)')
    max_pattern = re.compile(r'(.+)-max\[(ms|J)\]: ([\d.]+)')
    min_pattern = re.compile(r'(.+)-min\[(ms|J)\]: ([\d.]+)')
    median_pattern = re.compile(r'(.+)-median\[(ms|J)\]: ([\d.]+)')

    # Initialize variables to store the parsed data
    data = {}
    nkernels = None

    with open(file_path, 'r') as file:
        for line in file:
            # Check for frequency
            nkernels_match = nkernels_pattern.match(line)
            if nkernels_match:
                nkernels = int(nkernels_match.group(1))
                if nkernels not in data:
                    data[nkernels] = {'n_kernels': nkernels}
                continue

            # Check for frequency type
            type_match = type_pattern.match(line)
            if type_match:
                current_type = type_match.group(1).lower()  # 'App', 'Kernel' or 'Phase'
                continue               

            # Extract and store data
            if nkernels is not None:
                prefix = f'{current_type}_'
                
                energy_sample_before_match = energy_sample_before_pattern.match(line)
                if energy_sample_before_match:
                    energy_sample_before = energy_sample_before_match.group(1)
                    data[nkernels][f'{prefix}energy_sample_before'] = float(energy_sample_before)
                    continue
                
                energy_sample_after_match = energy_sample_after_pattern.match(line)
                if energy_sample_after_match:
                    energy_sample_after = energy_sample_after_match.group(1)
                    data[nkernels][f'{prefix}energy_sample_after'] = float(energy_sample_after)
                    continue
                
                energy_sample_delta_match = energy_sample_delta_pattern.match(line)
                if energy_sample_delta_match:
                    energy_sample_delta = energy_sample_delta_match.group(1)
                    data[nkernels][f'{prefix}energy_sample_delta'] = float(energy_sample_delta)
                    continue
                
                energy_sample_time_match = energy_sample_time_pattern.match(line)
                if energy_sample_time_match:
                    energy_sample_time = energy_sample_time_match.group(1)
                    data[nkernels][f'{prefix}energy_sample_time'] = float(energy_sample_time)
                    continue 
                
                time_value_match = time_value_pattern.match(line)
                kernel_value_match = kernel_value_pattern.match(line)
                device_energy_value_match = device_energy_value_pattern.match(line)
                host_energy_value_match = host_energy_value_pattern.match(line)
                freq_change_time_overhead_match = freq_change_time_overhead_pattern.match(line)
                freq_change_device_energy_overhead_match = freq_change_device_energy_overhead_pattern.match(line)
                freq_change_host_energy_overhead_match = freq_change_host_energy_overhead_pattern.match(line)

                # For total-time, device-energy, and host-energy
                for match, metric in zip([time_value_match, kernel_value_match, device_energy_value_match, host_energy_value_match, freq_change_time_overhead_match, freq_change_device_energy_overhead_match, freq_change_host_energy_overhead_match], 
                                         ['total_time', 'kernel_time', 'device_energy', 'host_energy', 'freq_change_time_overhead', 'freq_change_device_energy_overhead', 'freq_change_host_energy_overhead']):
                    if match:
                        # Extract other statistics
                        for _ in range(5):
                            stat_line = next(file)
                            avg_match = avg_pattern.match(stat_line)
                            if avg_match:
                                data[nkernels][f'{prefix}{metric}_Average'] = float(avg_match.group(3))
                            stdev_match = stdev_pattern.match(stat_line)
                            if stdev_match:
                                data[nkernels][f'{prefix}{metric}_Stdev'] = float(stdev_match.group(3))
                            max_match = max_pattern.match(stat_line)
                            if max_match:
                                data[nkernels][f'{prefix}{metric}_Max'] = float(max_match.group(3))
                            min_match = min_pattern.match(stat_line)
                            if min_match:
                                data[nkernels][f'{prefix}{metric}_Min'] = float(min_match.group(3))
                            median_match = median_pattern.match(stat_line)
                            if median_match:
                                data[nkernels][f'{prefix}{metric}_Median'] = float(median_match.group(3))
    
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