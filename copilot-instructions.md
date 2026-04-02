
---

## Step-by-Step Plan for Your Coding Agent

**GOAL**: Implement hierarchical two-phase trajectory generation using a single diffusion model, by using per-feature-group noise levels during training and inference, and designing a noise schedule accordingly. Phase 1 should plan robot root x, y, z, yaw, object position and rotation. Phase 2 should plan joints and body pose conditioned on the planned robot root and object trajectories.

**Step 1:** Extend discrete_diffusion.py to work with per-feature noise levels. Ensure that the indexing and shapes are consistent

**Step 2:** Add hierarchical noise level sampling to dfot_trajectory.py. This should ensure that root features (delta_xy, delta_yaw) are less noisy than joints/pose features.

**Step 3:** Wire hierarchical noise into the forward method of DFoTTrajectory
In the forward method of DFoTTrajectory in models/dfot_trajectory.py, find the block that calls _get_training_noise_levels. Add a config flag to switch between standard and hierarchical.

**Step 4:** Fix loss weighting for (B,T,D) shaped k, ensure the new shape does not raise errors in dfot_trajectory.py and discrete_diffusion.py. Do NOT reshape the noise levels to (B,T,1) as this will break the per-feature noise. Instead, ensure that the loss is computed correctly with the (B,T,D) noise levels.

**Step 5:** Structured inference scheduling matrix, which ensures root and object features are denoised first (maybe finish at step ~50% of total), and joints and pose features next (finish at 100%).

**Step 6:** Add config options for hierarchical noise levels and scheduling to the config files. Ensure that these options are properly read and used in the code.