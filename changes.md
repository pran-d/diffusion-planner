1) SWE changes: 
    <!-- Trimming and padding in FlexibleWindowDataset -->
    Inference speed-up in dfot_trajectory (no_grad, etc),
    Auto-computation of stitch steps during inference
2) Added state blending (CFG) in DiscreteDiffusion
3) Added group-wise input embedding in DiT1D
<!-- 4) EMA weights for model training and inference -->
<!-- 5) Physics-based clamping for CFG -->
6) Variable state history conditioning 
7) Auxiliary (physics-based) losses
8) Seeded training