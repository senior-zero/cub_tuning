#!/bin/env python3

import os
import time
import subprocess
from tqdm import tqdm

bin_dir = 'build/bin'

def tune(result_dir): 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    benches = []

    for filename in os.scandir(bin_dir):
        if filename.is_file():
            bench_path = filename.path
            bench_name = os.path.basename(bench_path)
            result_path = os.path.join(result_dir, bench_name + ".json")

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

