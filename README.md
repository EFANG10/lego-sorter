# LEGO Sorting Machine Synthetic Dataset Generator

A high-performance synthetic data generator built using **LDraw** and **LDView** to create training datasets for computer vision LEGO sorting machines.

---

## Architecture for LEGO Sorting Machine AI

A production sorting machine uses a two-stage computer vision pipeline:
1. **Model 1: Localizer / Detector (YOLOv8/v11, SSD, or OpenCV Contour Crop)**:
   Detects where the brick is located on the conveyor belt or chute, extracts a bounding box, and crops the brick.
2. **Model 2: Part Classifier (ResNet, MobileNetV3, EfficientNet, or ViT)**:
   Takes the cropped brick image and classifies it into its exact LEGO part number (`3001`, `3004`, `3020`, etc.).

This generator produces the training dataset for **Model 2 (Part Classification)**, matching the exact format expected by PyTorch `torchvision.datasets.ImageFolder` and TensorFlow `image_dataset_from_directory`.

---

## Dataset Directory Structure

```text
dataset/
├── 3001/                  # LEGO Part Number (e.g. 2x4 Brick)
│   ├── 3001_000001.jpg
│   ├── 3001_000002.jpg
│   └── ... (2,000+ images)
├── 3004/                  # LEGO Part Number (e.g. 1x2 Brick)
│   ├── 3004_000001.jpg
│   └── ... (2,000+ images)
└── <part_number>/
    └── ...
```

---

## Key Features

1. **High-Speed Multi-Process Rendering**:
   - Generates **~50 images/second** on modern multi-core CPUs (~40 seconds per 2,000 images).
2. **Realistic 3D Conveyor Poses & Tumbling Angles**:
   - **65% Canonical Resting Poses**: Bricks resting flat on studs, upside down, or on their sides, with 360° in-plane yaw and realistic tilt perturbations.
   - **35% Full 3D Tumbling Poses**: Uniform random 3D rotations from SO(3) to capture parts dropping down chutes or landing at angles.
   - **Overhead Camera Angles**: Random camera elevations ($35^\circ$ to $85^\circ$) and azimuths ($0^\circ$ to $360^\circ$).
3. **Authentic LEGO Color Palette**:
   - Integrates official LDraw color definitions (`LDConfig.ldr`, over 320 colors) with realistic weighting for high-volume production colors (Red, Blue, Yellow, Black, White, Light/Dark Bluish Gray, Tan, Lime, etc.).
4. **Sorting Machine Background Simulation**:
   - Matte black / dark gray rubber conveyor belts.
   - Clean white / off-white acrylic or 3D-printed sorting trays.
   - Industrial blue/green sorting chutes.
   - Sensor noise, subtle lighting changes, and motion blur simulation.
5. **Resume Capability**:
   - Safely resumes interrupted jobs without re-rendering existing images.

---

## Quick Start

### 1. Render the TOP 1,000 Most Common LEGO Parts (Recommended)
Ranked by frequency across all official LEGO sets in history (from Rebrickable dataset analysis):
```bash
./render_top_1000.py
```
Or directly via Python:
```bash
.venv/bin/python3 generate_dataset.py --top-1000 --num-images 2000
```
* **Target**: 1,000 parts $\times$ 2,000 images = 2,000,000 images
* **Disk Space**: **~19.7 GB** (comfortably fits on your drive)
* **Time**: **~12 hours** (finishes overnight!)
* **Resume**: Automatically skips any parts or images you have already generated.

### 2. Render EVERY Possible LEGO Part (All 23,800+ Official Parts)
```bash
./render_all_lego_parts.py
```
Or directly with Python:
```bash
.venv/bin/python3 generate_dataset.py --all-parts --num-images 2000
```
*Note: You can stop execution anytime with `Ctrl+C`. The script has automatic resume support and tracks completed parts in `dataset/dataset_manifest.json`.*

### 2. Render Batches / Slices of the Entire Library
```bash
# Render the first 100 parts from the full library:
.venv/bin/python3 generate_dataset.py --all-parts --num-images 2000 --limit 100

# Start from a specific part ID alphabetically:
.venv/bin/python3 generate_dataset.py --all-parts --num-images 2000 --start-from 3001
```

### 3. Dynamic Multi-Color Backgrounds & Anti-Overfitting
Every image randomly generates a unique background from the entire color wheel:
- Vibrant solid HSV hues across all $360^\circ$
- Dual-color linear lighting gradients
- Textured conveyor surface grain
- Industrial neutral tones (rubber black, slate gray, sorting white)
- Random 24-bit RGB brick colors mixed with official LEGO colors

This forces the neural network (CNN / ViT) to learn **geometric shape features (studs, tubes, walls, holes, aspect ratio)** rather than memorizing colors.

### 2. Generate for All Common LEGO Parts
A curated list of 40+ common bricks, plates, slopes, tiles, and technic pieces is provided in `common_parts.txt`:
```bash
./run_generation.sh --parts-file common_parts.txt --num-images 2000
```

### 3. Verify Dataset Integrity
```bash
.venv/bin/python3 verify_dataset.py
```

---

## CLI Options Reference (`generate_dataset.py`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--parts` | None | List of part numbers to render (e.g. `--parts 3001 3004`) |
| `--parts-file` | `common_parts.txt` | Path to a text file containing part IDs |
| `--num-images` | `2000` | Number of images to render per part |
| `--resolution` | `224` | Image width & height in pixels (224x224 standard) |
| `--output-dir` | `dataset` | Output directory |
| `--format` | `jpg` | Image format (`jpg` or `png`) |
| `--bg-mode` | `mixed` | Background: `mixed`, `conveyor_black`, `tray_white`, `random_noise`, `transparent` |
| `--no-augment` | False | Disables lighting, noise, and motion blur augmentations |
| `--workers` | CPU count | Number of parallel render workers |

---

## Training Example (PyTorch)

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('dataset', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

print(f"Classes: {dataset.classes}")
print(f"Total training images: {len(dataset)}")
```
