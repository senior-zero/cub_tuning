#!/bin/env python3

import os
import numpy as np
import statistics
import itertools
from nvbench_json import reader
from scipy.stats import mannwhitneyu


Ts = ['I32', 'I64', 'I16', 'I8', 'I128']
OffsetTs = ['I32', 'I64']
ProblemSizes = ['28', '24', '20', '16']
result_dir = 'build/result'
results = {}


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


for root, folders, files in os.walk(result_dir):
    for file in files:
        if not file.endswith('.json'):
            continue

        if root not in results:
            results[root] = {}

        algorithm = get_algorithm_name(file)

        if algorithm not in results[root]:
            results[root][algorithm] = {'base': '', 'variants': []}

        if file.endswith('.base.json'):
            results[root][algorithm]['base'] = os.path.join(root, file)
        else:
            results[root][algorithm]['variants'].append(
                os.path.join(root, file))


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
    alpha = 0.05

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
                    # H0: the distribution underlying `ref_samples` is not stochastically greater
                    # H1: the distribution underlying `ref_samples` is stochastically greater
                    _, p = mannwhitneyu(
                        ref_samples, cmp_samples, alternative='greater')

                    ref_median = statistics.median(ref_samples)
                    cmp_median = statistics.median(cmp_samples)

                    diff = cmp_median - ref_median
                    frac_diff = diff / ref_median

                    if p < alpha:
                        # Reject H0
                        stat.append(frac_diff * 100)

    return stat


for case in results:
    for algorithm in results[case]:
        data = results[case][algorithm]
        base = data['base']

        magic_value = 424242424242

        best_min_variant = base
        best_avg_variant = base
        best_min = magic_value
        best_avg = magic_value

        for variant in data['variants']:
            try:
                frac_diffs = compare(base, variant)
                if len(frac_diffs):
                    variant_min = min(frac_diffs)
                    variant_avg = sum(frac_diffs) / len(frac_diffs)
                    if variant_min < best_min:
                        best_min = variant_min
                        best_min_variant = variant
                    if variant_avg < best_avg:
                        best_avg = variant_avg
                        best_avg_variant = variant
            except Exception:
                pass

        if best_min != magic_value:
            print("{} ({}): min={} ({})".format(
                algorithm, case, best_min, best_min_variant))
