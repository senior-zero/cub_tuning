#!/bin/env python3

import os
import math
from nvbench_json import reader

result_dir = 'build/result'
results = {}

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
            results[root][algorithm]['variants'].append(os.path.join(root, file))


def version_tuple(v):
    return tuple(map(int, (v.split("."))))


all_devices = []
config_count = 0
unknown_count = 0
failure_count = 0
pass_count = 0


def find_matching_bench(needle, haystack):
    for hay in haystack:
        if hay["name"] == needle["name"] and hay["axes"] == needle["axes"]:
            return hay
    return None


def format_int64_axis_value(axis_name, axis_value, axes):
    axis = next(filter(lambda ax: ax["name"] == axis_name, axes))
    axis_flags = axis["flags"]
    value = int(axis_value["value"])
    if axis_flags == "pow2":
        value = math.log2(value)
        return "2^%d" % value
    return "%d" % value


def format_float64_axis_value(axis_name, axis_value, axes):
    return "%.5g" % float(axis_value["value"])


def format_type_axis_value(axis_name, axis_value, axes):
    return "%s" % axis_value["value"]


def format_string_axis_value(axis_name, axis_value, axes):
    return "%s" % axis_value["value"]


def format_axis_value(axis_name, axis_value, axes):
    axis = next(filter(lambda ax: ax["name"] == axis_name, axes))
    axis_type = axis["type"]
    if axis_type == "int64":
        return format_int64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "float64":
        return format_float64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "type":
        return format_type_axis_value(axis_name, axis_value, axes)
    elif axis_type == "string":
        return format_string_axis_value(axis_name, axis_value, axes)


def format_duration(seconds):
    if seconds >= 1:
        multiplier = 1.0
        units = "s"
    elif seconds >= 1e-3:
        multiplier = 1e3
        units = "ms"
    elif seconds >= 1e-6:
        multiplier = 1e6
        units = "us"
    else:
        multiplier = 1e6
        units = "us"
    return "%0.3f %s" % (seconds * multiplier, units)


def format_percentage(percentage):
    # When there aren't enough samples for a meaningful noise measurement,
    # the noise is recorded as infinity. Unfortunately, JSON spec doesn't
    # allow for inf, so these get turned into null.
    if percentage is None:
        return "inf"
    return "%0.2f%%" % (percentage * 100.0)


def compare_benches(ref_benches, cmp_benches, threshold):
    stat = []

    for cmp_bench in cmp_benches:
        ref_bench = find_matching_bench(cmp_bench, ref_benches)
        if not ref_bench:
            continue

        device_ids = cmp_bench["devices"]
        axes = cmp_bench["axes"]
        ref_states = ref_bench["states"]
        cmp_states = cmp_bench["states"]

        axes = axes if axes else []

        headers = [x["name"] for x in axes]
        colalign = ["center"] * len(headers)

        headers.append("Ref Time")
        colalign.append("right")
        headers.append("Ref Noise")
        colalign.append("right")
        headers.append("Cmp Time")
        colalign.append("right")
        headers.append("Cmp Noise")
        colalign.append("right")
        headers.append("Diff")
        colalign.append("right")
        headers.append("%Diff")
        colalign.append("right")
        headers.append("Status")
        colalign.append("center")

        for device_id in device_ids:

            rows = []
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

                def lookup_summary(summaries, tag):
                    return next(filter(lambda s: s["tag"] == tag, summaries), None)

                cmp_time_summary = lookup_summary(cmp_summaries, "nv/cold/time/gpu/mean")
                ref_time_summary = lookup_summary(ref_summaries, "nv/cold/time/gpu/mean")
                cmp_noise_summary = lookup_summary(cmp_summaries, "nv/cold/time/gpu/stdev/relative")
                ref_noise_summary = lookup_summary(ref_summaries, "nv/cold/time/gpu/stdev/relative")

                # TODO: Use other timings, too. Maybe multiple rows, with a
                # "Timing" column + values "CPU/GPU/Batch"?
                if not all([cmp_time_summary,
                            ref_time_summary,
                            cmp_noise_summary,
                            ref_noise_summary]):
                    continue

                def extract_value(summary):
                    summary_data = summary["data"]
                    value_data = next(filter(lambda v: v["name"] == "value", summary_data))
                    assert(value_data["type"] == "float64")
                    return value_data["value"]

                cmp_time = extract_value(cmp_time_summary)
                ref_time = extract_value(ref_time_summary)
                cmp_noise = extract_value(cmp_noise_summary)
                ref_noise = extract_value(ref_noise_summary)

                # Convert string encoding to expected numerics:
                cmp_time = float(cmp_time)
                ref_time = float(ref_time)

                diff = cmp_time - ref_time
                frac_diff = diff / ref_time

                stat.append(frac_diff * 100)

                if ref_noise and cmp_noise:
                    ref_noise = float(ref_noise)
                    cmp_noise = float(cmp_noise)
                    min_noise = min(ref_noise, cmp_noise)
                elif ref_noise:
                    ref_noise = float(ref_noise)
                    min_noise = ref_noise
                elif cmp_noise:
                    cmp_noise = float(cmp_noise)
                    min_noise = cmp_noise
                else:
                    min_noise = None  # Noise is inf

            if len(rows) == 0:
                continue

    return stat


def compare(base, variant):
    ref_root = reader.read_file(base)
    cmp_root = reader.read_file(variant)

    global all_devices
    all_devices = cmp_root["devices"]

    return compare_benches(ref_root["benchmarks"], cmp_root["benchmarks"], 0.0)


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
            print("{} ({}): min={} ({})".format(algorithm, case, best_min, best_min_variant))

