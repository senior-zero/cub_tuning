#!/bin/env python3

import os
import numpy as np
import statistics
import importlib
from nvbench_json import reader
from tabulate import tabulate

compare_distributions = importlib. import_module('compare.mannwhitneyu')
distributions_are_different = compare_distributions.distributions_are_different 

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


def find_matching_bench(needle, haystack):
    for hay in haystack:
        if hay["name"] == needle["name"] and hay["axes"] == needle["axes"]:
            return hay
    return None


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


def compare(base, variant):
    ref_root = reader.read_file(base)
    cmp_root = reader.read_file(variant)

    ref_benches = ref_root["benchmarks"]
    cmp_benches = cmp_root["benchmarks"]

    stat = []
    for cmp_bench in cmp_benches:
        ref_bench = find_matching_bench(cmp_bench, ref_benches)
        if not ref_bench:
            continue

        axes = cmp_bench["axes"]
        ref_states = ref_bench["states"]
        cmp_states = cmp_bench["states"]

        axes = axes if axes else []

        for cmp_state in cmp_states:
            cmp_state_name = cmp_state["name"]
            ref_state = next(filter(lambda st: st["name"] == cmp_state_name,
                                    ref_states),
                             None)
            if not ref_state:
                continue

            axis_values = cmp_state["axis_values"]
            if not axis_values:
                axis_values = []

            cmp_summaries = cmp_state["summaries"]
            ref_summaries = ref_state["summaries"]

            if not ref_summaries or not cmp_summaries:
                continue

            ref_samples = parse_samples(ref_state)
            cmp_samples = parse_samples(cmp_state)

            if len(ref_samples) > 0:
                if len(cmp_samples) > 0:
                    if distributions_are_different(ref_samples, cmp_samples):
                        ref_median = statistics.median(ref_samples)
                        cmp_median = statistics.median(cmp_samples)

                        diff = cmp_median - ref_median
                        frac_diff = diff / ref_median

                        stat.append(frac_diff * 100)

    return stat


for T in Ts:
    for OffsetT in OffsetTs:
        speedups = {}

        for Elements in ProblemSizes:
            results = {}
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
                    else:
                        results[algorithm]['variants'].append(file)

            root = os.path.join(result_dir, 'tdp')
            root = os.path.join(root, T)
            root = os.path.join(root, OffsetT)
            root = os.path.join(root, Elements)

            for algorithm in results:
                data = results[algorithm]
                base = data['base']

                for variant in data['variants']:
                    try:
                        frac_diffs = compare(os.path.join(root, base), os.path.join(root, variant))
                        if len(frac_diffs):
                            if algorithm not in speedups:
                                speedups[algorithm] = {}
                            if variant not in speedups[algorithm]:
                                speedups[algorithm][variant] = {}
                            speedups[algorithm][variant][Elements] = frac_diffs

                    except Exception:
                        pass
        
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

