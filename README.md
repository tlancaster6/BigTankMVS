# BigTankMVS

Frame-by-frame multi-view stereo 3D reconstruction for synchronized multi-camera arrays. Given calibrated, synchronized video streams from three or more cameras, the pipeline produces per-frame 3D point clouds in world coordinates.

## Methodology

### Pipeline

1. **Frame extraction** — For every *n*-th frame (configurable), each video is decoded, spatially downsampled, and undistorted using the camera's intrinsic/distortion parameters.
2. **Dense feature matching** — All camera pairs are matched using one of three GPU-accelerated matchers provided by [Kornia](https://kornia.readthedocs.io/):
   - **LoFTR** — Dense, detector-free matcher. Enhanced with a roll-angle rectification step that pre-rotates images so the world X-axis aligns with the image horizontal, improving match quality for downward-looking camera arrays.
   - **DISK + LightGlue** — Sparse keypoint detection (DISK) followed by learned matching (LightGlue).
   - **DeDoDe + LightGlue** — Alternative detector/descriptor (DeDoDe) with LightGlue matching.
3. **Pairwise triangulation** — Matched 2D correspondences are triangulated via `cv2.triangulatePoints` (DLT). Points with reprojection error above a configurable threshold are discarded.
4. **Point cloud merging** — Per-pair clouds are concatenated and deduplicated with Open3D voxel downsampling.
5. **Output** — Each frame's merged cloud is saved as a PLY file.

### Calibration

Camera parameters are supplied in a TOML file containing, per camera: intrinsic matrix, distortion coefficients, and extrinsics (Rodrigues rotation vector + translation). Camera names in the TOML are matched to video filenames via a configurable regex.

## Project Structure

```
bigtankmvs/
  core.py        # MultiViewStereoPipeline — main reconstruction loop
  utils.py       # Interactive Open3D point-cloud sequence viewer
  benchmark.py   # Single-frame comparison of all three matchers
  debug.py       # Single-frame diagnostics with per-pair stats and visualizations
```

## Usage

```python
from bigtankmvs import MultiViewStereoPipeline, visualize_sequence

pipeline = MultiViewStereoPipeline(
    project_name="my_project",
    frame_step=1800,
    matcher="loftr",              # or "disk_lightglue", "dedode_lightglue"
    downsample_factor=0.5,
    reprojection_error_threshold=2.0,
    duplicate_distance_threshold=5.0,
)
pipeline.process()

# Play back the resulting point cloud sequence
visualize_sequence("data/my_project/point_clouds/")
```

## Installation

Create and activate a conda env with Python 3.12 (for Open3D compatibility):
```
conda create -n BigTankMVS python=3.12
conda activate BigTankMVS
```
Install PyTorch (modify for your system per https://pytorch.org/get-started/locally/) and verify GPU availability:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```
Install remaining dependencies:
```
pip3 install kornia opencv-python open3d numpy toml tqdm
```