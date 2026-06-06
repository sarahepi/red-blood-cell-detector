"""
analysis.py
───────────
Batch performance analysis:
  • Run the detector on every image in a folder (or a list of paths)
  • Compare watershed ON vs OFF for each image
  • Print a formatted table of results
  • Export results to CSV

Usage:
    python analysis.py                          # auto-generates a sample
    python analysis.py path/to/image.jpg        # single image
    python analysis.py path/to/folder/          # whole folder
"""

import os
import sys
import time
import csv
import glob
from rbc_detector import RBCDetector, PipelineConfig
from generate_sample import make_sample


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def collect_images(source: str) -> list[str]:
    """Return a list of image paths from a file or directory."""
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(source, ext)))
        return sorted(paths)
    return []


def analyse(image_paths: list[str], config: PipelineConfig = None) -> list[dict]:
    """
    For each image, run with and without watershed.
    Returns a list of result dicts.
    """
    cfg = config or PipelineConfig()
    rows = []

    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│           RBC DETECTOR — BATCH PERFORMANCE ANALYSIS             │")
    print("├──────────────┬────────────┬────────────┬────────────┬────────────┤")
    print("│ Image        │ WS=OFF cnt │ WS=OFF ms  │ WS=ON  cnt │ WS=ON  ms  │")
    print("├──────────────┼────────────┼────────────┼────────────┼────────────┤")

    for path in image_paths:
        det = RBCDetector(cfg)
        compare = det.compare_watershed(path)

        r_off = compare[False]
        r_on  = compare[True]
        name  = os.path.basename(path)[:13].ljust(13)

        print(f"│ {name} │ {r_off.cell_count:>10} │ {r_off.elapsed_ms:>9.1f}  │ "
              f"{r_on.cell_count:>10} │ {r_on.elapsed_ms:>9.1f}  │")

        rows.append({
            "image":           os.path.basename(path),
            "count_no_ws":     r_off.cell_count,
            "ms_no_ws":        round(r_off.elapsed_ms, 1),
            "count_with_ws":   r_on.cell_count,
            "ms_with_ws":      round(r_on.elapsed_ms, 1),
            "delta_count":     r_on.cell_count - r_off.cell_count,
        })

    print("└──────────────┴────────────┴────────────┴────────────┴────────────┘")

    # Totals
    if rows:
        avg_no  = sum(r["ms_no_ws"]   for r in rows) / len(rows)
        avg_ws  = sum(r["ms_with_ws"] for r in rows) / len(rows)
        print(f"\n  Average latency — no watershed : {avg_no:.1f} ms")
        print(f"  Average latency — watershed    : {avg_ws:.1f} ms")
        print(f"  Watershed overhead             : +{avg_ws - avg_no:.1f} ms / image\n")

    return rows


def export_csv(rows: list[dict], path: str = "analysis_results.csv"):
    """Write results to a CSV file."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Results exported → {os.path.abspath(path)}")


def print_limitations():
    """
    Honest discussion of the classical CV approach's limitations.
    This is expected in a university report.
    """
    lims = """
╔══════════════════════════════════════════════════════════════════╗
║  LIMITATIONS OF THE CLASSICAL CV APPROACH                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. LIGHTING SENSITIVITY                                         ║
║     Thresholding assumes a bimodal histogram. Uneven             ║
║     illumination (vignetting) causes the background to bleed     ║
║     into the cell intensity range → false negatives.             ║
║     Fix: histogram equalisation (CLAHE) before thresholding.     ║
║                                                                  ║
║  2. OVERLAPPING / CLUMPED CELLS                                  ║
║     Watershed helps but fails when >3 cells form dense           ║
║     clusters — the distance transform peaks merge.               ║
║     Fix: iterative seeded watershed or DL instance segmentation. ║
║                                                                  ║
║  3. CELL SIZE VARIABILITY                                        ║
║     Fixed area/circularity thresholds discard abnormal cells     ║
║     (e.g., sickle cells, macrocytes) which are diagnostically    ║
║     important.                                                   ║
║                                                                  ║
║  4. STAIN / MICROSCOPE DEPENDENCE                               ║
║     Parameters tuned for Giemsa-stained slides may fail on       ║
║     Wright's stain or phase-contrast images.                     ║
║                                                                  ║
║  5. NO GROUND TRUTH VALIDATION                                   ║
║     Without a labelled dataset we cannot report precision /      ║
║     recall. Manual counting comparison is needed.                ║
║                                                                  ║
║  6. PROCESSING TIME                                              ║
║     Watershed is O(n log n) in pixel count — large 4K images     ║
║     may take 500–1000 ms. GPU-accelerated OpenCV can help.       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(lims)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else None

    if source is None:
        # Generate a synthetic sample for demonstration
        sample = make_sample("sample_rbc.jpg")
        paths  = [sample]
    else:
        paths = collect_images(source)

    if not paths:
        print("No images found. Provide a valid file or folder path.")
        sys.exit(1)

    rows = analyse(paths)
    export_csv(rows)
    print_limitations()
