"""
analysis_simple.py
──────────────────
Minimal helper to compare watershed ON vs OFF on a single image.
Useful for debugging or demonstration.
"""

from rbc_detector import RBCDetector, PipelineConfig


def compare_image(image_path: str):
    """
    Compare results with and without watershed on ONE image.
    """
    cfg = PipelineConfig()
    detector = RBCDetector(cfg)

    results = detector.compare_watershed(image_path)

    r_off = results[False]
    r_on  = results[True]

    print("\n=== COMPARISON ===")
    print(f"Without Watershed : {r_off.cell_count} cells ({r_off.elapsed_ms:.1f} ms)")
    print(f"With Watershed    : {r_on.cell_count} cells ({r_on.elapsed_ms:.1f} ms)")
    print(f"Difference        : {r_on.cell_count - r_off.cell_count} cells\n")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analysis_simple.py image.jpg")
    else:
        compare_image(sys.argv[1])