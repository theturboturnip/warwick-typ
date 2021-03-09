#!/usr/bin/env python3

import argparse
import glob
import os.path
from pathlib import Path
import statistics
import subprocess
import re

def average_num(tests_folder, filepath: str):
    with open(tests_folder / filepath, "r") as f:
        nums = [
            float(line)
            for line in f.readlines()
        ]

    # Remove min/max
    nums.remove(max(nums))
    nums.remove(min(nums))

    # Return average
    return statistics.mean(nums)

def check_all_eq(files):
    if subprocess.run(["diff", "-q", "--from-file", *files], stdout=subprocess.DEVNULL).returncode == 0:
        return True
    return False

def comparison_results(tests_folder, config, cpu="cpu_old_fast", cuda="cuda"):
    bins_cpu = [
        tests_folder / f"bin_{config}_{cpu}_{i}.bin"
        for i in range(1, 5+1)
    ]
    if not check_all_eq(bins_cpu):
        print(f"Error - not all binaries for {config} under {cpu} are the same")
    bins_cuda = [
        tests_folder / f"bin_{config}_{cuda}_{i}.bin"
        for i in range(1, 5+1)
    ]
    if not check_all_eq(bins_cuda):
        print(f"Error - not all binaries for {config} under {cuda} are the same")

    # print(bins_cpu[0], bins_cuda[0])
    cmp_result_proc = subprocess.run(["./sim_cuda", "compare", bins_cpu[0], bins_cuda[0]], capture_output=True)
    cmp_result = cmp_result_proc.stdout.decode('ascii')
    sq_error_mean_re = re.compile(r'Sq\. Error Mean:\t\t\s*([^\s]+)')
    return [
        float(match.group(1))
        for match in sq_error_mean_re.finditer(cmp_result)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tests_folder", type=Path)

    args = parser.parse_args()

    text_files = set(glob.glob(str(args.tests_folder / "*.txt")))

    test_backends = ["cpu_old_fast", "cuda"]

    configs = {
        os.path.basename(t[:-len(f"_{b}.txt")])
        for b in test_backends
        for t in text_files if t.endswith(f"_{b}.txt")
    }
    print(f"Found {len(configs)} configs")

    cuda_speedups = {}

    files_per_config = {
        c: {
            b: f"{c}_{b}.txt"
            for b in test_backends
        }
        for c in configs
    }
    for config, results in files_per_config.items():
        time_cpu = average_num(args.tests_folder, results[test_backends[0]])
        time_cuda = average_num(args.tests_folder, results[test_backends[1]])

        cuda_speedup = time_cpu/time_cuda
        cuda_speedups[config] = cuda_speedup

    mean = statistics.mean(cuda_speedups.values())
    print(f"speedup mean  = {mean}\n")
    print(f"speedup stdev = {statistics.stdev(cuda_speedups.values(), mean)}\n")
    for config in sorted(configs):
        print(config, "\t", cuda_speedups[config])

    print("\n", "Sq. Mean Errors (u, v, p)")
    for config in sorted(configs):
        print(config, "\t", comparison_results(args.tests_folder, config))