# OBSERVATIONS FROM ABLATIONS

1. Training on the following hardware:
- RTX3070
- Karolina cluster

2. Trained using 4 motion references, each of which has been augmented through SBTO data density.

- Batch size of 1024: large enough for good gradients, small enough to allow few updates per epoch
- Diffusion transformer as it can learn higher frequency variations in data than a UNet
- Condition dropout with some probability: enables classifier-free guidance to make task conditioning stronger. Comes at the cost of physical consistency for highly OOD tasks.
- Noise added to state observations and goal vector, to make the autoregressive generation robust to small errors. The noise levels are tuned per feature group, based on an estimate of the physical error allowed as per the sensors.

3. Normalized goal vector, clipped goal magnitude: allows using turning and walking motions from the dataset for larger out-of-distribution distances (as it appears to be in-distribution). Without clipping, overfits to dropping box at dataset distance. With clipping, can reach longer distances.

4. 6D rotation representation: helps learning as it removes discontinuities in rotation vectors that are present in other representations such as quaternion and euler angles

5. Mirror symmetry augmentation: Allows the dataset to be doubled based on symmetry, enabling learning of diverse skills from sparse data in a physically consistent manner. 

6. Auxiliary physical losses: 
- smoothness loss penalizes frame-to-frame jitter and thus smoothens the generated trajectory slightly.
- grasp consistency loss ensures that the object-hand distance remains roughly constant (it is a loss, thus a soft constraint) as long as the robot is grasping the box.

7. State conditioning masking: randomly masks parts of the state input. This helps break the correlation between history and future, allowing different features in the state history to affect the conditioning of the model differently

8. Redundant features: obj_delta_xy and obj_z can be included to allow inpainting waypoints during inference, forcing the model to follow the desired object trajectory and denoise the other features given true values for this. Redundant features because it is easier to give object trajectory in a fixed reference frame. It ensures box end position convergence, at the cost of physical consistency.

9. In-betweening (using waypoints) and partial masking: During training, random features (say, object position) are "unmasked" such that the model has to learn to generate the rest of the features (say, robot base position, joints, etc), given the unmasked features as ground truth. Correspondingly, during inference, if we know the desired end location of the box, we can determine a trajectory for object position through heuristics or supervised learning, and then generate the other features given this as the true box trajectory. Thus, we can enforce hard constraints for the box position during inference, and expect the model to learn the physical consistency given the desired box position. 
