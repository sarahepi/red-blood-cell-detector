"""

Pipeline overview:
  1. Load image
  2. Convert to grayscale
  3. Gaussian blur  → noise reduction
  4. Thresholding   → binary mask
  5. Morphological ops (erosion / dilation / opening / closing)
  6. Distance transform + Watershed → separate touching cells
  7. Contour detection + shape filtering
  8. Count & visualise
"""

import cv2
import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Tous les paramètres ajustables regroupés dans un seul endroit."""
    blur_ksize: int          = 7      # Gaussian kernel (must be odd)
    threshold_value: int     = 128    # manual threshold (0–255)
    use_otsu: bool           = True   # override threshold_value with Otsu
    morph_kernel_size: int   = 3      # structuring element size
    morph_iterations: int    = 2      # erosion / dilation iterations
    min_cell_area: int       = 400    # px² — discard tiny blobs
    max_cell_area: int       = 8000   # px² — discard huge artefacts
    min_circularity: float   = 0.55   # 1.0 = perfect circle
    use_watershed: bool      = True   # toggle watershed separation
    dist_threshold: float    = 0.5    # fraction of max for sure-fg


@dataclass
class PipelineResult:
    
    original:      Optional[np.ndarray] = None
    gray:          Optional[np.ndarray] = None
    blurred:       Optional[np.ndarray] = None
    thresholded:   Optional[np.ndarray] = None
    morphed:       Optional[np.ndarray] = None
    distance:      Optional[np.ndarray] = None
    watershed_map: Optional[np.ndarray] = None
    final:         Optional[np.ndarray] = None
    cell_count:    int                  = 0
    elapsed_ms:    float                = 0.0
    contours:      List                 = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# CORE PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────

class RBCDetector:
    """
    Encapsulates the full classical CV pipeline for RBC detection.

    Each method corresponds to one logical step and can be called
    independently for interactive / GUI use.
    """

    def __init__(self, config: PipelineConfig = None):
        self.cfg = config or PipelineConfig()
        self.result = PipelineResult()

    # ── Step 1: Load ────────────────────────────────────────────────

    def load_image(self, path: str) -> np.ndarray:
        """
        Read an image from disk.
        OpenCV loads BGR by default; we keep that convention throughout
        and convert to RGB only when handing off to Matplotlib/Tkinter.
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        self.result.original = img.copy()
        return img

    # ── Step 2: Grayscale ────────────────────────────────────────────

    def to_gray(self, img: np.ndarray) -> np.ndarray:
        """
        Convert BGR → grayscale.
        RBCs appear as darker or lighter discs depending on staining;
        grayscale preserves luminance information needed for thresholding.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.result.gray = gray
        return gray

    # ── Step 3: Noise reduction ──────────────────────────────────────

    def blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Gaussian blur smooths high-frequency noise that would otherwise
        produce false edges / blobs during thresholding.
        Kernel size must be odd; larger → stronger smoothing.
        """
        k = self.cfg.blur_ksize
        if k % 2 == 0:
            k += 1                            # enforce oddness
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        self.result.blurred = blurred
        return blurred

    # ── Step 4: Thresholding ─────────────────────────────────────────

    def threshold(self, blurred: np.ndarray) -> np.ndarray:
        """
        Convert grayscale → binary mask.

        Otsu's method (default):
          Automatically picks the threshold that minimises intra-class
          variance — great for bimodal histograms (cells vs background).

        Manual threshold:
          Useful when lighting is very uniform and the user wants
          fine control via the GUI slider.
        """
        if self.cfg.use_otsu:
            _, thresh = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, thresh = cv2.threshold(
                blurred,
                self.cfg.threshold_value, 255,
                cv2.THRESH_BINARY_INV
            )
        self.result.thresholded = thresh
        return thresh

    # ── Step 5: Morphological operations ─────────────────────────────

    def morphology(self, thresh: np.ndarray) -> np.ndarray:
        """
        Four classical morphological operations applied in sequence:

        EROSION   — shrinks white regions; removes thin protrusions /
                    noise pixels that survived thresholding.

        DILATION  — expands white regions; closes small gaps inside cells.

        OPENING   = erosion then dilation → removes small bright specks
                    (noise) while preserving cell shapes.

        CLOSING   = dilation then erosion → fills small dark holes inside
                    cells caused by staining artefacts.
        """
        k = self.cfg.morph_kernel_size
        n = self.cfg.morph_iterations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        eroded   = cv2.erode    (thresh,  kernel, iterations=n)
        dilated  = cv2.dilate   (eroded,  kernel, iterations=n)
        opened   = cv2.morphologyEx(dilated, cv2.MORPH_OPEN,  kernel)
        closed   = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, kernel)

        self.result.morphed = closed
        return closed

    # ── Step 6a: Distance Transform ──────────────────────────────────

    def distance_transform(self, morphed: np.ndarray) -> np.ndarray:
        """
        For each foreground pixel, compute its Euclidean distance to the
        nearest background pixel.  Peaks in this map correspond to cell
        centres — they are far from any edge.

        Used by Watershed to seed the markers (sure foreground).
        """
        dist = cv2.distanceTransform(morphed, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        self.result.distance = dist
        return dist

    # ── Step 6b: Watershed ───────────────────────────────────────────

    def watershed(
        self,
        original: np.ndarray,
        morphed:  np.ndarray,
        dist:     np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Watershed algorithm separates touching / overlapping cells.

        Key idea — treat the inverted distance map as a topographic
        surface. 'Water' fills from seed peaks; watershed lines form at
        ridges between basins → those ridges become cell boundaries.

        Steps:
          1. sure_bg  = dilated morphed mask  (definitely background)
          2. sure_fg  = thresholded distance  (definitely cell centres)
          3. unknown  = sure_bg - sure_fg     (uncertain border pixels)
          4. Label connected components in sure_fg as markers
          5. Mark unknown region as 0 (let watershed decide)
          6. Apply cv2.watershed → overwrites marker boundaries with -1
        """
        # Sure background: one dilation pass on the whole mask
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg  = cv2.dilate(morphed, kernel, iterations=3)

        # Sure foreground: pixels far enough from any edge
        _, sure_fg = cv2.threshold(
            dist, self.cfg.dist_threshold * dist.max(), 255, 0
        )
        sure_fg  = np.uint8(sure_fg)

        # Unknown region
        unknown  = cv2.subtract(sure_bg, sure_fg)

        # Label markers — connected components in the sure-foreground
        _, markers = cv2.connectedComponents(sure_fg)
        markers   += 1            # background becomes 1 (not 0)
        markers[unknown == 255] = 0   # unknown → 0 for watershed

        # Run watershed on the colour image
        img_ws    = original.copy()
        markers   = cv2.watershed(img_ws, markers)

        # Draw watershed boundary lines (-1) in red
        img_ws[markers == -1] = [0, 0, 255]

        self.result.watershed_map = markers
        return markers, img_ws

    # ── Step 7: Contour detection & filtering ────────────────────────

    def detect_contours(
        self,
        markers:  Optional[np.ndarray],
        morphed:  np.ndarray,
        original: np.ndarray
    ) -> Tuple[np.ndarray, List, int]:
        """
        Find and filter contours representing individual RBCs.

        Without watershed: contours are extracted directly from the
        morphological mask.

        With watershed: each unique label (>1) is a separate cell;
        we re-binarise each label region and find its contour.

        Filtering criteria (configurable):
          • Area: discard dust (too small) and cell clusters (too large)
          • Circularity: 4π·Area / Perimeter² → RBCs are roughly circular
        """
        output = original.copy()
        valid_contours = []

        if self.cfg.use_watershed and markers is not None:
            # ── Watershed path ──────────────────────────────────────
            cell_mask = np.zeros(morphed.shape, dtype=np.uint8)
            for label in np.unique(markers):
                if label <= 1:          # 0 = unknown, 1 = background
                    continue
                mask = np.zeros(morphed.shape, dtype=np.uint8)
                mask[markers == label] = 255
                cnts, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in cnts:
                    if self._is_valid(c):
                        valid_contours.append(c)
        else:
            # ── Direct contour path ─────────────────────────────────
            cnts, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid_contours = [c for c in cnts if self._is_valid(c)]

        # Draw enclosing circles + index label for each detected cell
        for i, c in enumerate(valid_contours, start=1):
            (x, y), r = cv2.minEnclosingCircle(c)
            cx, cy, cr = int(x), int(y), int(r)
            cv2.circle(output, (cx, cy), cr, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 3,  (0, 0, 255), -1)  # centre dot
            cv2.putText(
                output, str(i), (cx - 8, cy - cr - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                cv2.LINE_AA
            )

        # Overlay total count
        count = len(valid_contours)
        label = f"RBC Count: {count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(output, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)
        cv2.putText(
            output, label, (14, 14 + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2, cv2.LINE_AA
        )

        self.result.final     = output
        self.result.contours  = valid_contours
        self.result.cell_count = count
        return output, valid_contours, count

    # ── Helper: shape filter ─────────────────────────────────────────

    def _is_valid(self, contour: np.ndarray) -> bool:
        """
        Reject contours that are clearly NOT individual RBCs.

        Circularity = 4π·A / P²
          = 1.0  → perfect circle
          ≈ 0.75 → slightly elongated
          < 0.5  → very irregular (likely noise or two merged cells)
        """
        area = cv2.contourArea(contour)
        if not (self.cfg.min_cell_area <= area <= self.cfg.max_cell_area):
            return False
        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return False
        circularity = (4 * np.pi * area) / (peri ** 2)
        return circularity >= self.cfg.min_circularity

    # ── Full pipeline (single call) ──────────────────────────────────

    def run(self, image_path: str) -> PipelineResult:
        """
        Execute the complete pipeline end-to-end and return a
        PipelineResult with every intermediate image and the final count.
        """
        t0 = time.perf_counter()

        img      = self.load_image(image_path)
        gray     = self.to_gray(img)
        blurred  = self.blur(gray)
        thresh   = self.threshold(blurred)
        morphed  = self.morphology(thresh)

        markers = None
        if self.cfg.use_watershed:
            dist    = self.distance_transform(morphed)
            markers, _ = self.watershed(img, morphed, dist)

        self.detect_contours(markers, morphed, img)

        self.result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return self.result

    # ── Comparison helper ────────────────────────────────────────────

    def compare_watershed(self, image_path: str) -> dict:
        """
        Run the pipeline twice — with and without watershed — and return
        both results for side-by-side analysis.
        """
        results = {}
        for ws in (False, True):
            self.cfg.use_watershed = ws
            self.result = PipelineResult()          # reset
            results[ws] = self.run(image_path)
        return results


# ─────────────────────────────────────────────────────────────────────
# STANDALONE CLI RUNNER
# ─────────────────────────────────────────────────────────────────────

def cli_run(image_path: str, config: PipelineConfig = None):
    """
    Run the detector from the command line, display results with
    Matplotlib, and print performance stats.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Matplotlib not installed — using cv2.imshow fallback.")
        plt = None

    cfg = config or PipelineConfig()
    det = RBCDetector(cfg)
    r   = det.run(image_path)

    print("=" * 60)
    print(f"  Image        : {os.path.basename(image_path)}")
    print(f"  Cells found  : {r.cell_count}")
    print(f"  Elapsed      : {r.elapsed_ms:.1f} ms")
    print(f"  Watershed    : {'ON' if cfg.use_watershed else 'OFF'}")
    print("=" * 60)

    steps = [
        ("Original",       cv2.cvtColor(r.original, cv2.COLOR_BGR2RGB)),
        ("Grayscale",      r.gray),
        ("Blurred",        r.blurred),
        ("Thresholded",    r.thresholded),
        ("Morphological",  r.morphed),
        ("Final Detection",cv2.cvtColor(r.final,    cv2.COLOR_BGR2RGB)),
    ]

    if plt:
        fig = plt.figure(figsize=(18, 7), facecolor="#0d0d0d")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.15)
        fig.suptitle(
            f"RBC Detector — {r.cell_count} cells detected  |  {r.elapsed_ms:.0f} ms",
            color="white", fontsize=14, fontweight="bold"
        )
        for idx, (title, img) in enumerate(steps):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color="#aaaaaa", fontsize=9)
            ax.axis("off")
        plt.savefig("rbc_result.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("  Result saved : rbc_result.png")
        plt.show()
    else:
        for title, img in steps:
            cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_rbc.jpg"
    cli_run(path)
