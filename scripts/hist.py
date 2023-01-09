#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.stats import shapiro 

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

from nvbench_json import reader


def extract_filename(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "filename", summary_data))
    assert(value_data["type"] == "string")
    return value_data["value"]


def extract_size(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "size", summary_data))
    assert(value_data["type"] == "int64")
    return int(value_data["value"])


def parse_samples_meta(filename, state):
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


def parse_samples(filename, state):
    sample_count, samples_filename = parse_samples_meta(filename, state)
    if not sample_count or not samples_filename:
        return []

    with open(samples_filename, "rb") as f:
        samples = np.fromfile(f, "<f4")

    assert (sample_count == len(samples))
    return samples


def parse_json(filename):
    json_root = reader.read_file(filename)

    for bench in json_root["benchmarks"]:
        for state in bench["states"]:
            samples = parse_samples(filename, state)
            if len(samples) == 0:
                continue

            alpha = 0.05
            stat, p = shapiro(samples)

            if p > alpha:
                print(filename, len(samples), "gaussian") # Fail to reject H0, looks Gaussian
            else:
                print(filename, len(samples), "not gaussian") # Reject H0, doesn\'t look Gaussian

            sns.histplot(data=samples)
            plt.show()


for root, folders, files in os.walk('build/result'):
    for file in files:
        path = os.path.join(root, file)

        if not path.endswith('.json'):
            continue

        if os.path.getsize(path) == 0:
            continue

        parse_json(path)

