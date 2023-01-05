#!/bin/env python3

import os
import time
import subprocess
from tqdm import tqdm

bin_dir = 'build/bin'
types = ['I32', 'I64', 'I16', 'I8', 'I128']
offset_types = ['I32', 'I64']
elements = ['28', '24', '20', '16']


def get_result_dir(result_dir_base, T, OffsetT, Elements):
    result_path = os.path.join(result_dir_base, T)
    result_path = os.path.join(result_path, OffsetT)
    result_path = os.path.join(result_path, Elements)
    return result_path


def get_result_path(result_dir, bench_name):
    return os.path.join(result_dir, bench_name + ".json")


def tune(result_dir_base): 
    benches = []

    for T in types:
        for OffsetT in offset_types:
            for Elements in elements:
                result_dir = get_result_dir(result_dir_base, T, OffsetT, Elements)

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                for filename in os.scandir(bin_dir):
                    if filename.is_file():
                        bench_path = filename.path
                        bench_name = os.path.basename(bench_path)
                        result_path = get_result_path(result_dir, bench_name)

                        if os.path.exists(result_path):
                            if os.path.isfile(result_path):
                                if os.path.getsize(result_path) > 0:
                                    continue

                        if not bench_name.startswith("bench.device"):
                            continue

                        benches.append(bench_path)

                for bench_path in tqdm(benches):
                    bench_name = os.path.basename(bench_path)
                    result_path = os.path.join(result_dir, bench_name + ".json")

                    cmdline = [bench_path, "--json", result_path, "--device", "0"]
                    cmdline = cmdline + ["-a", "T={}".format(T)]
                    cmdline = cmdline + ["-a", "OffsetT={}".format(OffsetT)]
                    cmdline = cmdline + ["-a", "Elements[pow2]={}".format(Elements)]
                    subprocess.run(cmdline, stdout=subprocess.DEVNULL)


def run_as_root(cmdline):
    if os.getlogin() != 'root':
        cmdline = ['sudo'] + cmdline

    if subprocess.run(cmdline, stdout=subprocess.DEVNULL).returncode != 0:
        raise Exception("Can't run as root: {}".format(cmdline))


def reset_clocks():
    run_as_root(['nvidia-smi', '-rgc'])
    run_as_root(['nvidia-smi', '-pm', '0'])


def fix_clocks():
    run_as_root(['nvidia-smi', '-pm', '1'])
    run_as_root(['nvidia-smi', '-lgc', 'tdp'])


fix_clocks()
tune('build/result/tdp')
reset_clocks()
tune('build/result/base')

