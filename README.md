# Diffusion Motion Planning Project

This folder contains a complete, standalone pipeline for training and evaluating diffusion-based motion planning models.

## 1. Setup & Data

### Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup
The training and inference scripts require a dataset in `.npz` format.

1.  **Clone or Copy Data**: Place your dataset folder (containing `.npz` trajectory files) into the `datasets/` directory.
    *   Example structure: `datasets/Box_Height_FoLM/*.npz`

2.  **Update Config**:
    Open `config/config.yaml` and update the `data` section to point to your dataset location.
    
    ```yaml
    data:
      dir_path: "./"                  # Base directory
      train_path: "datasets/Box_Height_FoLM" # Path relative to dir_path
    ```

    *   `dir_path`: The root for data lookups.
    *   `train_path`: The folder containing your specific dataset files.

## 2. Training a Model

To start training a new model, simply run:

```bash
python train.py
```

Settings:
*   **Hyperparameters**: Modified in `config/config.yaml`.
*   **Logs**: Saved to `runs/logs/`.
*   **Checkpoints**: Saved to `runs/checkpoints/`.

## 3. Inference & Visualization

The `inference.py` script allows you to generate trajectories, stitch multiple segments together (long-horizon rollout), and visualize the results.

### Basic Generation
Generate samples using a trained checkpoint:

```bash
python inference.py --epoch runs/checkpoints/model_1000.pth
```

### Stitched Rollout
To generate longer trajectories by autoregressively stitching segments:

```bash
python inference.py --epoch runs/checkpoints/model_1000.pth --stitch_steps 3
```

### Visualization
To view the generated trajectories in MuJoCo:

```bash
python inference.py --epoch runs/checkpoints/model_1000.pth --stitch_steps 3 --visualize
```

*   **Controls**:
    *   `SPACE`: Pause/Play
    *   `Right Arrow`: Step forward
    *   `Left Arrow`: Step backward
    *   `ESC`: Exit

*   **Note**: Visualization requires `mj_model.xml` to be present in the root directory or the dataset directory. A default `mj_model.xml` is included in this folder.
