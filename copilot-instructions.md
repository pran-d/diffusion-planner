# Plan: Unified Multi-Style Box Locomotion Model

Train a single diffusion model on mixed trajectories where style (pick-place, push, kick) is inferred **dynamically** from motion characteristics (e.g., end-effector contact patterns) rather than file selection via `chosen_tasks.yml`. The model conditions on auto-detected style to produce style-appropriate continuations.

## Steps

1. **Audit dataset pipeline**: Examine `train.py`, `config/config.yaml` → `paths.yaml` linking, and `config/configure.py` to understand current task-file selection in `chosen_tasks.yml`. Identify where to inject style-detection logic.

2. **Design motion-based style classifier**: 
   - Analyze trajectory features that distinguish styles (e.g., end-effector contact geometry with box, hand/foot positions, forces/torques if available, joint sequences).
   - Implement a heuristic or learned classifier to infer `style ∈ {pick-place, push, kick}` from a short window of motion (e.g., first 10-20 timesteps or contact peaks).
   - Document feature extraction requirements (which joint/task variables needed in dataset).

3. **Modify dataset loading** in `datasets/flexible_dataset.py`, `datasets/conditional_dataset.py`, and `datasets/buffer_dataset.py`:
   - Merge all three style datasets into a single data source (or keep separate but load all at train time).
   - Compute style label per trajectory/segment during dataset construction.
   - Return `(trajectory_tensor, style_id, style_features)` tuples.

4. **Route style conditioning through model**:
   - Embed style_id via `diffusion_forcing_transformer/embeddings.py` (e.g., learnable style embedding).
   - Pass style embedding to UNet/DIT in `models/dfot_trajectory.py` and `models/model.py` (crossAttn, concat, or FiLM conditioning).
   - Optionally, also condition on style_features (contact patterns) directly for fine-grained control.

5. **Update config and training**:
   - Modify `config/config.yaml` to accept unified dataset path(s) and style detection parameters.
   - Update `train.py` to handle mixed-style batches (no per-style file selection).
   - For inference, add command override (e.g., `--force_style=kick`) or auto-detect from context.

## Key Decisions & Considerations

1. **Style detection mechanism**:
   - Contact-based (recommended): end-effector in contact with box → pick-place/push vs kick (no contact). Within pick-place vs push, use height/hand position heuristics.
   - Feature-based: hand velocity + distance, foot contact, arm lift angle.
   - Learned: train a lightweight classifier on labeled subset upfront.

2. **Detection window**: Classify from first N timesteps, or use sliding window + majority vote? Pick one that's robust to partial observations at inference.

3. **Sequence length mismatch**: Do all styles have similar episode lengths? If push is longer, pad shorter styles or train with variable-length masking.

4. **Inference style control**: How does the user specify desired style at generation time?
   - Explicit: `--style=pick` flag.
   - Implicit: condition on initial observation (let model infer).
   - Hybrid: both options.

5. **Balancing**:
   - Will raw mixed data have equal style distribution? If not, oversample or weighted loss per style.
   - Monitor per-style metrics (separate validation sets) to detect mode collapse to dominant style.
