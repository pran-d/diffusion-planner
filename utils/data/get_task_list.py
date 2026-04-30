import json
import re
import yaml

def load_tasks(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def filter_by_regex(dataset_path, pattern, output_yaml):
    dataset = load_tasks(dataset_path)
    regex = re.compile(pattern)

    selected = []

    for entry in dataset:
        if regex.search(entry["task"]):
            selected.append(entry["original_folder"])

    # Save YAML
    with open(output_yaml, "w") as f:
        yaml.dump({"original_folders": selected}, f)

    return selected

# Usage
filter_by_regex(
    "tasks.yml",
    pattern=r"\\[pick]\\",
    output_yaml="filtered.yml"
)