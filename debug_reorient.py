import numpy as np

# Load inference script or just run it with a patch
import sys
import subprocess

out = subprocess.check_output(
    ["python", "inference_mg.py", "--task", "reorient", "--task_params", "0.0", "0.0", "0.0"],
    text=True
)
lines = out.split('\n')
for line in lines:
    if "Goal world position" in line or "New initial obj pos" in line or "Shifted" in line or "Updated goal_local" in line:
        print(line)
    if "Loaded 1 initial conditions" in line:
        print("Loaded!")
