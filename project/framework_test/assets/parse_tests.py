#!/usr/bin/env python3

import argparse
import glob
import os.path
from pathlib import Path
import statistics
import subprocess
import re
import pandas

def average_num(tests_folder, filepath: str, include_ticks: bool):
    with open(tests_folder / filepath, "r") as f:
        if include_ticks:
            lines = f.readlines()
            nums = [
                float(line)
                for line in lines[::2]
            ]
            ticks = [
                int(line)
                for line in lines[1::2]
            ]
        else:
            nums = [
                float(line)
                for line in f.readlines()
            ]
            ticks = [0] * len(nums)

    if ticks != [ticks[0]] * len(ticks):
        raise RuntimeError(f"not all tick values for {filepath} equal - {ticks}")

    # Remove min/max
    nums.remove(max(nums))
    nums.remove(min(nums))

    # Return average
    return statistics.mean(nums), ticks[0]

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

    # Files might not exist
    bins_cpu_that_exist = [b for b in bins_cpu if os.path.isfile(b)]
    bins_cuda_that_exist = [b for b in bins_cuda if os.path.isfile(b)]
    if not bins_cpu_that_exist or not bins_cuda_that_exist:
        raise RuntimeError(f"Error - config {config} has no binaries")

    # print(bins_cpu[0], bins_cuda[0])
    cmp_result_proc = subprocess.run(["./sim_cuda", "compare", bins_cpu_that_exist[0], bins_cuda_that_exist[0]], capture_output=True)
    cmp_result = cmp_result_proc.stdout.decode('ascii')
    sq_error_mean_re = re.compile(r'Sq\. Error Mean:\s*([^\s]+)')
    return [
        float(match.group(1))
        for match in sq_error_mean_re.finditer(cmp_result)
    ]

def residual(tests_folder, config, backend):
    bins = [
        tests_folder / f"bin_{config}_{backend}_{i}.bin"
        for i in range(1, 5+1)
    ]
    bins_that_exist = [b for b in bins if os.path.isfile(b)]
    if not bins_that_exist:
        raise RuntimeError(f"Error - config {config} has no binaries")
    bin_path = bins_that_exist[0]

    # Don't need to use the exact fluid.json, only thing that's different between them is N which isn'tused here
    residual_result_proc = subprocess.run(["./sim_cuda", "residual", "fluid_100.json", bin_path], capture_output=True)
    residual_results = residual_result_proc.stdout.decode('ascii').splitlines()
    # print("'", residual_result, "'")
    # print(residual_result_proc.args)
    return [float(l) for l in residual_results]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tests_folder", type=Path)
    parser.add_argument("--cpu-type", default="cpu", type=str)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--no-ticks", dest="include_ticks", action="store_false")
    
    args = parser.parse_args()

    text_files = set(glob.glob(str(args.tests_folder / "*.txt")))

    test_backends = [args.cpu_type, "cuda"]

    configs = {
        os.path.basename(t[:-len(f"_{b}.txt")])
        for b in test_backends
        for t in text_files if t.endswith(f"_{b}.txt")
    }
    print(f"Found {len(configs)} configs")

    datafile = pandas.DataFrame(
        index=sorted(configs),
        columns=[
            "valid",
            "fluid iters", "input file",
            "avg_time_cpu", "avg_time_cuda", "ticks_cpu", "ticks_cuda",
            "sq. error mean (u)", "sq. error mean (v)",
            "min_u_cpu", "max_u_cpu",
            "min_u_cuda", "max_u_cuda",
            "min_v_cpu", "max_v_cpu",
            "min_v_cuda", "max_v_cuda",

            "sq. error mean (p)", "sq. error mean (p relative)",
            "min_p_cpu", "max_p_cpu",
            "min_p_cuda", "max_p_cuda",

            "residual_cuda", "residual_cpu"
        ])
    for config in configs:
        if "fluid_100" in config:
            datafile.loc[config]["fluid iters"] = 100
        elif "fluid_200" in config:
            datafile.loc[config]["fluid iters"] = 200
        elif "fluid_300" in config:
            datafile.loc[config]["fluid iters"] = 300
        elif "fluid_accurate" in config:
            datafile.loc[config]["fluid iters"] = 1000
        else:
            raise RuntimeError(f"config {config} did not have a valid fluid-type")

        if "initial_aca" in config:
            datafile.loc[config]["input file"] = "initial_aca"
        elif "initial_circles" in config:
            datafile.loc[config]["input file"] = "initial_circles_large"
        else:
            datafile.loc[config]["input file"] = "unk"

    cuda_speedups = {}

    files_per_config = {
        c: {
            b: f"{c}_{b}.txt"
            for b in test_backends
        }
        for c in configs
    }
    for config, results in files_per_config.items():
        try:
            time_cpu, ticks_cpu = average_num(args.tests_folder, results[test_backends[0]], args.include_ticks)
            time_cuda, ticks_cuda = average_num(args.tests_folder, results[test_backends[1]], args.include_ticks)

            datafile.loc[config]["avg_time_cpu"] = time_cpu
            datafile.loc[config]["ticks_cpu"] = ticks_cpu
            datafile.loc[config]["avg_time_cuda"] = time_cuda
            datafile.loc[config]["ticks_cuda"] = ticks_cuda

            cuda_speedup = time_cpu/time_cuda
            cuda_speedups[config] = cuda_speedup

            datafile.loc[config]["valid"] = 1
        except RuntimeError as e:
            print(e)

            datafile.loc[config]["valid"] = 0
            cuda_speedups[config] = 0


    mean = statistics.mean(cuda_speedups.values())
    print(f"speedup mean  = {mean}\n")
    # print(f"speedup stdev = {statistics.stdev(cuda_speedups.values(), mean)}\n")
    for config in sorted(configs):
        print(config, "\t", cuda_speedups[config])

    print("\n", "Sq. Mean Errors (u, v, p)")
    for config in sorted(configs):
        try:
            sq_error_mean = comparison_results(args.tests_folder, config, cpu=args.cpu_type)
            print(config, "\t", sq_error_mean)
            if len(sq_error_mean) != 4:
                raise RuntimeError("unexpected length of sq_error_mean")
            datafile.loc[config]["sq. error mean (u)"] = sq_error_mean[0]
            datafile.loc[config]["sq. error mean (v)"] = sq_error_mean[1]
            datafile.loc[config]["sq. error mean (p)"] = sq_error_mean[2]
            datafile.loc[config]["sq. error mean (p relative)"] = sq_error_mean[3]

            if float("nan") in sq_error_mean:
                datafile.loc[config]["valid"] = 0

        except RuntimeError as e:
            print(e)

            datafile.loc[config]["valid"] = 0
            cuda_speedups[config] = 0

    print("\n", "Residuals")
    for config in sorted(configs):
        try:
            residual_cpu, min_u_cpu, max_u_cpu, min_v_cpu, max_v_cpu, min_p_cpu, max_p_cpu = residual(args.tests_folder, config, test_backends[0])
            residual_cuda, min_u_cuda, max_u_cuda, min_v_cuda, max_v_cuda, min_p_cuda, max_p_cuda = residual(args.tests_folder, config, test_backends[1])
            print(config, "\t", test_backends[0], "\t", residual_cpu, "\t", min_p_cpu, "\t", max_p_cpu)
            print(config, "\t", test_backends[1], "\t", residual_cuda, "\t", min_p_cuda, "\t", max_p_cuda)

            datafile.loc[config]["residual_cpu"] = residual_cpu
            datafile.loc[config]["min_u_cuda"] = min_u_cuda
            datafile.loc[config]["max_u_cuda"] = max_u_cuda
            datafile.loc[config]["min_v_cuda"] = min_v_cuda
            datafile.loc[config]["max_v_cuda"] = max_v_cuda
            datafile.loc[config]["min_p_cuda"] = min_p_cuda
            datafile.loc[config]["max_p_cuda"] = max_p_cuda

            datafile.loc[config]["residual_cuda"] = residual_cuda
            datafile.loc[config]["min_u_cpu"] = min_u_cpu
            datafile.loc[config]["max_u_cpu"] = max_u_cpu
            datafile.loc[config]["min_v_cpu"] = min_v_cpu
            datafile.loc[config]["max_v_cpu"] = max_v_cpu
            datafile.loc[config]["min_p_cpu"] = min_p_cpu
            datafile.loc[config]["max_p_cpu"] = max_p_cpu
        except RuntimeError as e:
            print(e)

            datafile.loc[config]["valid"] = 0
            cuda_speedups[config] = 0

    if args.output_csv:
        with open(args.output_csv, "w") as f:
            datafile.to_csv(f)