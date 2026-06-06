# RBC Detector — Automatic Red Blood Cell Detection & Counting
### Classical Computer Vision · OpenCV only · 
---

## Project Structure

```
rbc_detector/
├── rbc_detector.py      ← Core pipeline (load → detect → count)
├── gui.py               ← Tkinter GUI application
├── analysis.py          ← Batch performance analysis & comparison
├── generate_sample.py   ← Synthetic test-image generator
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## Installation

**Python 3.10+ required.**

```bash
# 1. Clone / unzip the project folder
cd rbc_detector

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### A) GUI Application (recommended)
```bash
python gui.py
```
1. Click **Open Image** → select a microscopy image (JPG/PNG/TIFF)
2. Adjust sliders (blur, threshold, morphological kernel, area/circularity filters)
3. Toggle **Watershed** on/off
4. Click **▶ Run Pipeline**
5. Navigate intermediate steps with the tab buttons at the top
6. Click **⚖ Compare ±Watershed** to see both results side-by-side
7. Click **💾 Save Result** to export the annotated image

### B) Command-line (single image)
```bash
python rbc_detector.py path/to/image.jpg
```
Displays a 6-panel Matplotlib figure and saves `rbc_result.png`.

### C) Batch analysis
```bash
# Single image
python analysis.py path/to/image.jpg



### D) Generate a synthetic test image
```bash
python generate_sample.py
# → saves sample_rbc.jpg in the current directory
```

---



## Dependencies

| Library | Version | Role |
|---------|---------|------|
| `opencv-python` | ≥ 4.8 | All CV operations |
| `numpy` | ≥ 1.24 | Array maths |
| `Pillow` | ≥ 10.0 | Tkinter image bridge |
| `matplotlib` | ≥ 3.7 | CLI visualisation (optional) |
| `tkinter` | stdlib | GUI (ships with Python) |

---


