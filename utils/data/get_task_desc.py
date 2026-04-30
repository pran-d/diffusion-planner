import json
import yaml

# Load your JSONL dataset
def load_tasks(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load YAML list
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fetch_tasks(dataset_path, yaml_path):
    dataset = load_tasks(dataset_path)
    yml = load_yaml(yaml_path)
    folders = set(yml)

    results = []
    for entry in dataset:
        if "original_folder" in entry.keys() and entry["original_folder"] in folders:
            results.append({
                "original_folder": entry["original_folder"],
                "task": entry["task"]
            })

    return results

# Usage
results = fetch_tasks("./test_datasets/tasks.yml", "./test_datasets/chosen_tasks.yml")

for r in results:
    print(r)