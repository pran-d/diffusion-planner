import mujoco
import os

model_path = "scene_with_ghost.xml"

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"Successfully loaded model: {model_path}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
except Exception as e:
    print(f"Failed to load model: {model_path}")
    print(e)
