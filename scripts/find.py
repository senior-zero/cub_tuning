#!/bin/env python3

import os
import json

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
        if not file.endswith(".json"):
            continue

        if root not in results:
            results[root] = {}

        algorithm = get_algorithm_name(file)

        if algorithm not in results[root]:
            results[root][algorithm] = []

        results[root][algorithm].append(os.path.join(root, file))

print(json.dumps(results, indent=2))

