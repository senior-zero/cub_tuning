#!/bin/env python3

import os
import time
import subprocess
import itertools
from tqdm import tqdm


bin_dir = 'build/bin'
cases = itertools.product(['I32', 'I64', 'I16', 'I8', 'I128'],  # T
                          ['I32', 'I64'],  # OffsetT
                          ['28', '24', '20', '16'])  # Elements


def get_result_dir(result_dir_base, T, OffsetT, Elements):
    result_path = os.path.join(result_dir_base, T)
    result_path = os.path.join(result_path, OffsetT)
    result_path = os.path.join(result_path, Elements)
    return result_path


def get_result_path(result_dir, bench_name):
    return os.path.join(result_dir, bench_name + ".json")


def is_bench(bench_path):
    return bench_path.startswith('bench.device')


def is_base_bench(bench_path):
    return bench_path.endswith('.base')


def can_skip_bench(bench_name, result_path):
    if bench_name.endswith('.base'):
        return False

    if not os.path.exists(result_path):
        return False

    if not os.path.isfile(result_path):
        return False

    if os.path.getsize(result_path) > 0:
        return True

    return False


def run_bench(result_dir_base, bench_path, base_elapsed={}):
    elapsed = {}

    for T, OffsetT, Elements in cases:
        result_dir = get_result_dir(
            result_dir_base, T, OffsetT, Elements)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        bench_name = os.path.basename(bench_path)
        result_path = os.path.join(result_dir, bench_name + ".json")

        if can_skip_bench(bench_name, result_path):
            continue

        cmd = [bench_path, "--json", result_path]
        cmd = cmd + ["--device", "0"]
        cmd = cmd + ["-a", "T={}".format(T)]
        cmd = cmd + ["-a", "OffsetT={}".format(OffsetT)]
        cmd = cmd + \
            ["-a", "Elements[pow2]={}".format(Elements)]

        lbl = "{}.{}.{}".format(T, OffsetT, Elements)

        timeout = None
        if lbl in base_elapsed:
            timeout = base_elapsed[lbl] * 4

        try:
            begin = time.time()
            subprocess.run(
                cmd, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elapsed[lbl] = time.time() - begin
        except subprocess.TimeoutExpired:
            tqdm.write("variant {} ({}/{}/{}) reached timeout {}s".format(
                bench_path, T, OffsetT, Elements, timeout))

    return elapsed


def get_benches():
    bases = []
    variants = []

    for filename in os.scandir(bin_dir):
        if filename.is_file():
            bench_path = filename.path
            bench_name = os.path.basename(bench_path)

            if not is_bench(bench_name):
                continue

            if is_base_bench(bench_name):
                bases.append(bench_path)
            else:
                variants.append(bench_path)

    return bases, variants


def get_base_elapsed(timeouts, bench_path):
    ref = {}
    for base in timeouts:
        if bench_path.startswith(base):
            ref = timeouts[base]

    return ref


def tune(result_dir_base):
    timeouts = {}
    bases, variants = get_benches()

    print('run bases:')
    for bench_path in tqdm(bases):
        timeouts[bench_path.removesuffix('.base')] = run_bench(
            result_dir_base, bench_path)

    print('run variants:')
    for bench_path in tqdm(variants):
        ref = get_base_elapsed(timeouts, bench_path)
        run_bench(result_dir_base, bench_path, ref)


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
