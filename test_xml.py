import mujoco
xml_path = "unitree_g1/mj_model.xml" # or wherever the model is
print("Parsing XML...")
# Let's find the exact path using glob
import glob
xml_paths = glob.glob("**/mj_model*.xml", recursive=True)
for p in xml_paths:
    print(p)
