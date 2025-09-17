import argparse
import subprocess
import sys

def get_core_frequencies(query_freq_path):
    """
    Runs the query_freq application and parses its output to get
    a list of available core frequencies.
    """
    print("INFO: Querying available GPU frequencies...")
    try:
        result = subprocess.run(
            [query_freq_path],
            capture_output=True,
            text=True,
            check=True
        )
    except FileNotFoundError:
        print(f"ERROR: '{query_freq_path}' not found. Please ensure it is compiled and the path is correct.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Running '{query_freq_path}' failed with exit code {e.returncode}.", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    core_freqs = []
    for line in result.stdout.splitlines():
        if line.startswith("core_freq:"):
            # Split by space and convert to int, ignoring the "core_freq:" part
            try:
                core_freqs = [int(freq) for freq in line.split()[1:]]
                break
            except ValueError:
                print(f"ERROR: Could not parse core frequencies from line: '{line}'", file=sys.stderr)
                sys.exit(1)

    if not core_freqs:
        print("ERROR: Could not find 'core_freq:' line in the output of query_freq.", file=sys.stderr)
        print(f"Stdout from query_freq:\n{result.stdout}", file=sys.stderr)
        sys.exit(1)
    
    core_freqs.sort()
    print(f"INFO: Found {len(core_freqs)} core frequencies: {core_freqs}")
    return core_freqs

def run_freq_overhead_test(overhead_test_path, num_runs, num_kernels, polling_time, freq1, freq2, output_file=None):
    """
    Runs the freq_overhead_test with the given parameters.
    """
    print("-" * 80)
    print(f"INFO: Running benchmark with polling_time={polling_time}us, freq1={freq1}MHz, freq2={freq2}MHz")
    
    args = [
        overhead_test_path,
        str(num_runs),
        str(polling_time),
        str(num_kernels),
        str(freq1),
        str(freq2)
    ]

    def execute(handle):
        # Using Popen to stream output in real-time
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Read and print output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)
                if handle:
                    handle.write(line + '\n')

        rc = process.poll()
        if rc != 0:
            print(f"WARNING: '{' '.join(args)}' exited with non-zero code {rc}.", file=sys.stderr)

    try:
        execute(output_file)

    except FileNotFoundError:
        print(f"ERROR: '{overhead_test_path}' not found. Please ensure it is compiled and the path is correct.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run frequency change overhead benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of runs for the benchmark.'
    )
    parser.add_argument(
        '--num-kernels',
        type=int,
        default=256,
        help='Number of kernels to launch in each run.'
    )
    parser.add_argument(
        '--query-freq-path',
        type=str,
        default='./build/samples/query_freq',
        help='Path to the query_freq executable. This is required to have the frequency available on the target device'
    )
    parser.add_argument(
        '--overhead-script-path',
        type=str,
        default='./build/samples/freq_overhead_test',
        help='Path to the freq_overhead_test executable.'
    )
    parser.add_argument(
        '--output-log',
        type=str,
        default='.',
        help='Path to save the output log file (freq_overhead_test.log).'
    )
    
    args = parser.parse_args()

    core_freqs = get_core_frequencies(args.query_freq_path)
    
    if len(core_freqs) < 2:
        print("ERROR: Need at least two core frequencies to run the benchmark.", file=sys.stderr)
        sys.exit(1)

    polling_times_us = [0, 100, 500]

    output_log_path = f"{args.output_log}/freq_overhead_test.log"
    print(f"INFO: Saving output to {output_log_path}")

    with open(output_log_path, 'w') as f:
        for polling_time in polling_times_us:
            # Iterate from min to max frequency
            min_freq_index = 0
            max_freq_index = len(core_freqs) - 1

            while min_freq_index < max_freq_index:
                freq1 = core_freqs[min_freq_index]
                freq2 = core_freqs[max_freq_index]

                run_freq_overhead_test(
                    args.overhead_script_path,
                    args.num_runs,
                    args.num_kernels,
                    polling_time,
                    freq1,
                    freq2,
                    output_file=f
                )

                min_freq_index += 1
                max_freq_index -= 1

if __name__ == '__main__':
    main()
