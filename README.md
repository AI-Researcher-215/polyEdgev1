# PolyEdge V1.0

This folder contains the code release for the PolyEdge paper.

The goal is to provide a reproducible path for:

- running generation with the proposed PolyEdge schedule
- visualizing results with the provided tools

## Included

- `inference/`
  - `channel_processor.py`: Channel-time modulation logic for PolyEdge signals
  - `pipeline.py`: Single-ControlNet inference pipeline with channel-time modulation
  - `merge_image.py`: Result visualization helper

## Recommended environment

Python 3.11 with a CUDA-enabled PyTorch install.

Install dependencies with:

```powershell
pip install -r requirements.txt
```

## Expected local layout

The default scripts expect the following structure:

```text
PolyEdge/
  models/
    stable-diffusion-v1-5/
    sd_controlnet_canny/
  inference/
```

You can override most paths from the command line.

## Quick start

### Basic usage example

```python
from inference.pipeline import MultiControlNetPipeline3Chan
import cv2
import numpy as np

# Load your PolyEdge control image (RGB format)
poly_edge_image = cv2.imread("poly_edge.png")
poly_edge_image = cv2.cvtColor(poly_edge_image, cv2.COLOR_BGR2RGB)

# Initialize pipeline
pipeline = MultiControlNetPipeline3Chan(
    sd_path="./models/stable-diffusion-v1-5",
    cn_path="./models/sd_controlnet_canny",
    r_range=(0.0, 1.0),    # Outline channel active range
    g_range=(0.35, 0.7),   # Detail channel active range
    b_range=(0.0, 0.2),    # Background channel active range
)

# Run generation
result = pipeline(
    poly_edge_image=poly_edge_image,
    prompt="a high quality photo, realistic, natural lighting",
    num_inference_steps=35,
    guidance_scale=7.5,
    seed=42,
)

# Save result
result.save("output.png")
```