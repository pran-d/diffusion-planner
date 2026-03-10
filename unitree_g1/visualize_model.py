import mujoco
import mujoco.viewer
import time
import numpy as np

model_path = "scene_with_ghost_29dof_new.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f"Loaded model with nq={model.nq}, nv={model.nv}")

# Print joint names to understand the structure
print("Joint names:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"Joint {i}: {name}")

print(model.nq)

# Example: Set some joint values
# You can access joints by name using model.joint(name).qposadr
# Let's set the ghost robot to be slightly offset in position so it's visible
# ghost_root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ghost_floating_base_joint")
# if ghost_root_joint_id != -1:
#     qpos_adr = model.jnt_qposadr[ghost_root_joint_id]
#     # Move ghost robot 1 meter to the right
#     data.qpos[qpos_adr + 1] += 1.0 
#     print(f"Moved ghost robot to y={data.qpos[qpos_adr + 1]}")

model.opt.gravity = (0, 0, 0)  # Disable gravity so ghost doesn't fall through floor

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Apply some movement to joints for visualization
        # Example: Sine wave on the first actuated joint of the main robot
        # Assuming the first few qpos after root are joints
        # data.qpos[7] = 0.5 * np.sin(time.time())

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
