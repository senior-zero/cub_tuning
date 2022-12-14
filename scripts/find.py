#!/bin/env python3

import os
import argparse
import numpy as np
import importlib
from tqdm import tqdm
from nvbench_json import reader
from tabulate import tabulate
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing import current_process

parser = argparse.ArgumentParser()
parser.add_argument('--compare', type=str, default='compare.mannwhitneyu')
parser.add_argument('--center', type=str, default='center.median')

args = parser.parse_args()

compare_distributions = importlib.import_module(args.compare)
distributions_are_different = compare_distributions.distributions_are_different 

center_of_distribution = importlib.import_module(args.center)
center = center_of_distribution.center

Ts = ['I32', 'I64', 'I16', 'I8', 'I128']
OffsetTs = ['I32', 'I64']
ProblemSizes = ['16', '20', '24', '28']
result_dir = 'build/result'


def get_element_to_weight_mapping():
    np.random.seed(42)
    weights = np.sort(np.random.dirichlet([x * 4 for x in range(1, len(ProblemSizes) + 1)], size=1))[0]

    element_to_weight = {}
    for EID, Elements in enumerate(ProblemSizes):
        element_to_weight[Elements] = weights[EID]

    return element_to_weight


ElementToWeight = get_element_to_weight_mapping()


def extract_filename(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "filename", summary_data))
    assert (value_data["type"] == "string")
    return value_data["value"]


def extract_size(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "size", summary_data))
    assert (value_data["type"] == "int64")
    return int(value_data["value"])


def get_algorithm_name(file):
    attrs = file.split('.')
    last_name_attr = 0
    for aid, attr in enumerate(attrs):
        if attr == 'json':
            continue

        if 'base' == attr:
            continue

        if '_' not in attr:
            last_name_attr = max(last_name_attr, aid)

    return '.'.join(attrs[2:last_name_attr+1])


def parse_samples_meta(state):
    summaries = state["summaries"]
    if not summaries:
        return None, None

    summary = next(filter(lambda s: s["tag"] == "nv/json/bin:nv/cold/sample_times",
                          summaries),
                   None)
    if not summary:
        return None, None

    sample_filename = extract_filename(summary)
    sample_count = extract_size(summary)
    return sample_count, sample_filename


def parse_samples(state):
    sample_count, samples_filename = parse_samples_meta(state)
    if not sample_count or not samples_filename:
        return []

    with open(samples_filename, "rb") as f:
        samples = np.fromfile(f, "<f4")

    assert (sample_count == len(samples))
    return samples


def read_samples(json_path):
    result = {}

    try:
        meta = reader.read_file(json_path)
        benches = meta["benchmarks"]

        for bench in benches:
            bench_name = bench["name"]
            result[bench_name] = {}
            states = bench["states"]

            for state in states:
                state_name = state["name"]
                result[bench_name][state_name] = parse_samples(state)
    except Exception:
        pass

    return result


def compare(base_samples, root, base, variant):
    stat = []

    base_path = os.path.join(root, base)
    variant_path = os.path.join(root, variant)

    all_base_samples = base_samples[base_path]['samples']
    all_base_centers = base_samples[base_path]['center']

    try:
        cmp_root = reader.read_file(variant_path)
        cmp_benches = cmp_root["benchmarks"]

        for cmp_bench in cmp_benches:
            bench_name = cmp_bench["name"]
            if bench_name not in all_base_samples:
                continue

            states = cmp_bench["states"]

            for state in states:
                state_name = state["name"]

                if state_name not in all_base_samples[bench_name]:
                    continue

                ref_samples = all_base_samples[bench_name][state_name]
                cmp_samples = parse_samples(state)

                if len(ref_samples) > 0:
                    if len(cmp_samples) > 0:
                        if distributions_are_different(ref_samples, cmp_samples):
                            ref_median = all_base_centers[bench_name][state_name]
                            cmp_median = center(cmp_samples)

                            diff = cmp_median - ref_median
                            frac_diff = diff / ref_median

                            stat.append(frac_diff * 100)
                        else:
                            pass
    except Exception:
        pass

    return variant, stat


for T in Ts:
    pool = Pool(8)

    for OffsetT in OffsetTs:
        speedups = {}

        for Elements in ProblemSizes:
            results = {}
            base_samples = {}
            case_dir = os.path.join(result_dir, 'tdp')
            case_dir = os.path.join(case_dir, T)
            case_dir = os.path.join(case_dir, OffsetT)
            case_dir = os.path.join(case_dir, Elements)

            for root, folders, files in os.walk(case_dir):
                for file in files:
                    if not file.endswith('.json'):
                        continue

                    algorithm = get_algorithm_name(file)

                    if algorithm not in results:
                        results[algorithm] = {'base': '', 'variants': []}

                    if file.endswith('.base.json'):
                        results[algorithm]['base'] = file
                        base_path = os.path.join(root, file)
                        samples = read_samples(base_path)
                        centers = {}

                        for bench in samples:
                            centers[bench] = {}
                            for state in samples[bench]:
                                centers[bench][state] = center(samples[bench][state])

                        base_samples[base_path] = {'samples': samples,
                                                   'center': centers} 
                    else:
                        results[algorithm]['variants'].append(file)

            print("processing {}/{}/{}:".format(T, OffsetT, Elements))
            for algorithm in results:
                data = results[algorithm]
                base = data['base']

                # for variant in data['variants']:
                    # v, frac_diffs = compare(case_dir, base, variant)
                closure = partial(compare, base_samples, case_dir, base)
                variants = data['variants']
                for variant, frac_diffs in tqdm(pool.imap_unordered(closure, variants), total=len(variants)):
                    if len(frac_diffs):
                        if algorithm not in speedups:
                            speedups[algorithm] = {}
                        if variant not in speedups[algorithm]:
                            speedups[algorithm][variant] = {}
                        speedups[algorithm][variant][Elements] = frac_diffs
        
        table = []
        print("T={}; OffsetT={};".format(T, OffsetT))
        for algorithm in speedups:
            scores = {}

            for variant in speedups[algorithm]:
                score = 0.0

                if len(speedups[algorithm][variant]) != len(ProblemSizes):
                    continue

                for Elements in speedups[algorithm][variant]:
                    for Speedup in speedups[algorithm][variant][Elements]:
                        score = score + Speedup * ElementToWeight[Elements]

                scores[variant] = score

            if len(scores):
                best_variant = min(scores, key=scores.get)
                best_variant_short_name = best_variant.replace("bench.device.", "").replace(algorithm + '.', '').replace('.json', '')
                speedup = speedups[algorithm][best_variant]
                table.append([algorithm, best_variant_short_name, scores[best_variant], speedup['16'], speedup['20'], speedup['24'], speedup['28']])
        print(tabulate(table))

