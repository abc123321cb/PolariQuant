# PolariQuant

PolariQuant is a Python tool for quantitative analysis of rotating-analyzer transmission polariscope data. Given a TIFF image stack captured at multiple analyzer angles, it fits a sinusoidal model to recover modulation strength, fit quality, and in-plane principal-axis orientation maps.

## What it does

- Loads an aligned, cropped TIFF stack (RGB or grayscale)
- ROI analysis: mean intensity vs analyzer angle + cosine model fit
- Full-field analysis: sliding-window cosine fitting to generate maps
  - Modulation amplitude (Imod)
  - Fit quality (R²)
  - Orientation (θ) from fit phase (180° axis ambiguity)
- Optional RGB analysis: per-channel maps (R/G/B) plus a combined orientation estimate
- Reproduces figures through an interactive menu or CLI flags

## Input data

PolariQuant expects a TIFF stack ordered by analyzer angle.

Default angle set in code:
- 0°, 15°, 30°, 45°, 60°, 75°, 90° (7 frames)

Accepted stack shapes:
- RGB: `(N, H, W, 3)`
- Grayscale: `(N, H, W)`

If your acquisition uses different angles or a different number of frames, update `ANGLES_DEG` in `PolariQuant.py`.

## Installation

Recommended: use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install numpy matplotlib tifffile
```

## Quick start

### 1) Run with a preconfigured dataset (optional)

`PolariQuant.py` includes dataset presets (`fork`, `plastic`) that point to default paths like:
- `data/fork_aligned_cropped.tif`
- `data/stressed_plastic_aligned_cropped.tif`

Run using the preset path:
```bash
python PolariQuant.py --dataset fork
```

Or run using preset ROIs but your own TIFF path:
```bash
python PolariQuant.py path/to/your_stack.tif --dataset fork
```

### 2) Run on your own TIFF stack (no preset ROIs)

```bash
python PolariQuant.py path/to/your_stack.tif
```

This will run the analysis and then open an interactive figure menu. ROI figures (1 and 2) require ROI coordinates (see below).

### 3) Generate specific figures from the command line

Examples:
```bash
python PolariQuant.py --dataset fork --figure 3
python PolariQuant.py --dataset fork --figure 1,2,6
python PolariQuant.py path/to/your_stack.tif --figure all
```

Change sliding window size (default is 10x10):
```bash
python PolariQuant.py --dataset fork --figure 3 --box 15
```

## Figures

Figure numbers match the options in the script:

1. ROI intensity vs angle  
2. ROI data and cosine fits  
3. Gray modulation (Imod) heat map  
4. Gray R² heat map  
5. Gray orientation angle map  
6. Gray orientation quiver map  
7. RGB normalized modulation heat maps  
8. RGB R² heat maps  
9. RGB orientation quiver maps  
10. Combined RGB orientation quiver map  
11. Gray vs combined RGB quiver comparison  
12. Gray vs combined RGB orientation difference map  
13. Histogram of gray vs combined RGB orientation difference  

## Defining ROIs

ROIs are stored in `DATASET_CONFIGS` inside `PolariQuant.py` as `(r0, r1, c0, c1)` using Python slice conventions:
- rows: `r0:r1`
- cols: `c0:c1`

To enable ROI figures (1 and 2) for your own stack:
- Add a new dataset entry in `DATASET_CONFIGS`, or
- Reuse an existing preset and edit its ROI coordinates

## Notes and troubleshooting

- If you see a warning about the number of slices not matching `ANGLES_DEG`, either update `ANGLES_DEG` or re-export your stack with the expected ordering.
- Grayscale-only stacks skip RGB analysis automatically.
- The plotting code currently uses `plt.show()` windows. If you want saved outputs, add `plt.savefig(...)` in the figure functions.
