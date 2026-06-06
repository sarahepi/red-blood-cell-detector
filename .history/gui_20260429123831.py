"""
╔══════════════════════════════════════════════════════════════════════╗
║   RBC DETECTOR — Tkinter GUI                                        ║
║   A professional, dark-themed desktop interface for interactive     ║
║   exploration of each pipeline stage.                               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import sys

# Import our pipeline
from rbc_detector import RBCDetector, PipelineConfig, PipelineResult

# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (dark scientific UI)
# ─────────────────────────────────────────────────────────────────────
BG      = "#0e1117"
PANEL   = "#161b22"
ACCENT  = "#00d084"
ACCENT2 = "#58a6ff"
TEXT    = "#e6edf3"
MUTED   = "#7d8590"
BORDER  = "#21262d"
ERROR   = "#f85149"
WARN    = "#d29922"

FONT_MONO = ("Consolas", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_HEAD = ("Segoe UI Semibold", 11)
FONT_TITL = ("Segoe UI Light",   20)


# ─────────────────────────────────────────────────────────────────────
# CUSTOM WIDGETS
# ─────────────────────────────────────────────────────────────────────

class FlatButton(tk.Button):
    """Flat, accent-coloured button with hover effect."""

    def __init__(self, parent, **kwargs):
        colour  = kwargs.pop("colour",  ACCENT)
        fg      = kwargs.pop("fg",      "#000000")
        super().__init__(
            parent,
            bg=colour, fg=fg,
            activebackground=colour,
            activeforeground=fg,
            relief="flat", bd=0,
            padx=12, pady=6,
            cursor="hand2",
            font=FONT_BODY,
            **kwargs
        )
        self._colour = colour
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        # Lighten on hover by blending with white
        self.config(bg=self._lighten(self._colour))

    def _on_leave(self, _):
        self.config(bg=self._colour)

    @staticmethod
    def _lighten(hex_col: str, factor: float = 1.15) -> str:
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"


class StyledScale(tk.Scale):
    """Dark-themed horizontal slider."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            orient="horizontal",
            bg=PANEL, fg=TEXT,
            activebackground=ACCENT,
            highlightthickness=0,
            troughcolor=BORDER,
            sliderrelief="flat",
            font=FONT_BODY,
            **kwargs
        )


class LogPanel(tk.Frame):
    """Scrollable monospace log output."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=PANEL, **kwargs)
        self._text = tk.Text(
            self, bg=PANEL, fg=ACCENT, font=FONT_MONO,
            state="disabled", wrap="word",
            bd=0, highlightthickness=0,
            insertbackground=ACCENT
        )
        sb = tk.Scrollbar(self, command=self._text.yview,
                          bg=BORDER, troughcolor=PANEL)
        self._text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)

    def log(self, msg: str, colour: str = None):
        self._text.config(state="normal")
        tag = f"col_{id(colour)}"
        if colour:
            self._text.tag_configure(tag, foreground=colour)
        self._text.insert("end", msg + "\n", tag if colour else "")
        self._text.see("end")
        self._text.config(state="disabled")

    def clear(self):
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────

class RBCApp(tk.Tk):

    STEPS = [
        "original", "gray", "blurred",
        "thresholded", "morphed", "final"
    ]
    STEP_LABELS = [
        "Original", "Grayscale", "Blurred",
        "Thresholded", "Morphological", "Final Detection"
    ]

    def __init__(self):
        super().__init__()
        self.title("RBC Detector  ·  Classical Computer Vision")
        self.configure(bg=BG)
        self.geometry("1380x820")
        self.minsize(1100, 700)

        # ── State ────────────────────────────────────────────────────
        self.image_path:   str                    = ""
        self.result:       PipelineResult | None   = None
        self.compare_res:  dict                   = {}
        self.current_step: int                    = 0
        self._photo_ref:   ImageTk.PhotoImage | None = None

        # Config vars (bound to widgets)
        self.v_threshold  = tk.IntVar(value=128)
        self.v_blur       = tk.IntVar(value=7)
        self.v_morph_k    = tk.IntVar(value=3)
        self.v_morph_iter = tk.IntVar(value=2)
        self.v_min_area   = tk.IntVar(value=400)
        self.v_max_area   = tk.IntVar(value=8000)
        self.v_min_circ   = tk.DoubleVar(value=0.55)
        self.v_dist_thr   = tk.DoubleVar(value=0.5)
        self.v_otsu       = tk.BooleanVar(value=True)
        self.v_watershed  = tk.BooleanVar(value=True)

        self._build_ui()

    # ═════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────
        bar = tk.Frame(self, bg=PANEL, height=56)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(bar, text="🔬  RBC Detector", font=FONT_TITL,
                 bg=PANEL, fg=TEXT).pack(side="left", padx=20, pady=10)
        tk.Label(bar, text="Classical OpenCV Pipeline",
                 font=FONT_BODY, bg=PANEL, fg=MUTED).pack(side="left")

        # Version badge
        tk.Label(bar, text="v1.0  |  No DL", font=FONT_MONO,
                 bg=ACCENT, fg="#000").pack(side="right", padx=20, ipadx=8, ipady=4)

        # ── Main layout ──────────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        # Left panel (controls)
        left = tk.Frame(main, bg=PANEL, width=290)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        self._build_controls(left)

        # Centre (image canvas)
        centre = tk.Frame(main, bg=BG)
        centre.pack(side="left", fill="both", expand=True)
        self._build_canvas(centre)

        # Right panel (log)
        right = tk.Frame(main, bg=PANEL, width=270)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        self._build_log(right)

    # ── Controls ─────────────────────────────────────────────────────

    def _build_controls(self, parent):
        pad = {"padx": 14, "pady": 4}

        def section(text):
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", **pad)
            tk.Label(parent, text=text.upper(), font=("Segoe UI Semibold", 8),
                     bg=PANEL, fg=MUTED).pack(anchor="w", **pad)

        tk.Label(parent, text="CONTROLS", font=FONT_HEAD,
                 bg=PANEL, fg=TEXT).pack(pady=(14, 6))

        # File
        section("Image")
        FlatButton(parent, text="📂  Open Image",
                   command=self._open_image, colour=ACCENT2, fg="#000"
                   ).pack(fill="x", padx=14, pady=4)
        self.lbl_file = tk.Label(parent, text="No file selected",
                                 font=FONT_MONO, bg=PANEL, fg=MUTED,
                                 wraplength=240, justify="left")
        self.lbl_file.pack(anchor="w", padx=14, pady=(0, 4))

        # Threshold
        section("Thresholding")
        tk.Checkbutton(parent, text="Use Otsu (auto)",
                       variable=self.v_otsu, bg=PANEL, fg=TEXT,
                       selectcolor=PANEL, activebackground=PANEL,
                       activeforeground=TEXT,
                       command=self._toggle_otsu).pack(anchor="w", padx=14)
        tk.Label(parent, text="Manual threshold:", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        self.sl_thresh = StyledScale(parent, from_=0, to=255,
                                     variable=self.v_threshold)
        self.sl_thresh.pack(fill="x", padx=14)
        self.sl_thresh.config(state="disabled")

        # Blur
        section("Gaussian Blur")
        tk.Label(parent, text="Kernel size (odd):", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=1, to=21, variable=self.v_blur
                    ).pack(fill="x", padx=14)

        # Morphology
        section("Morphological Ops")
        tk.Label(parent, text="Kernel size:", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=1, to=15, variable=self.v_morph_k
                    ).pack(fill="x", padx=14)
        tk.Label(parent, text="Iterations:", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=1, to=6, variable=self.v_morph_iter
                    ).pack(fill="x", padx=14)

        # Filters
        section("Contour Filters")
        tk.Label(parent, text="Min area (px²):", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=50, to=3000, variable=self.v_min_area
                    ).pack(fill="x", padx=14)
        tk.Label(parent, text="Max area (px²):", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=1000, to=20000, variable=self.v_max_area
                    ).pack(fill="x", padx=14)
        tk.Label(parent, text="Min circularity:", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=0.1, to=1.0, resolution=0.05,
                    variable=self.v_min_circ).pack(fill="x", padx=14)

        # Watershed
        section("Watershed")
        tk.Checkbutton(parent, text="Enable Watershed separation",
                       variable=self.v_watershed, bg=PANEL, fg=TEXT,
                       selectcolor=PANEL, activebackground=PANEL,
                       activeforeground=TEXT).pack(anchor="w", padx=14)
        tk.Label(parent, text="Distance threshold:", font=FONT_BODY,
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)
        StyledScale(parent, from_=0.1, to=0.9, resolution=0.05,
                    variable=self.v_dist_thr).pack(fill="x", padx=14)

        # Action buttons
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14, pady=8)
        FlatButton(parent, text="▶  Run Pipeline",
                   command=self._run_pipeline).pack(fill="x", padx=14, pady=3)
        FlatButton(parent, text="⚖  Compare ± Watershed",
                   command=self._run_compare,
                   colour="#6e40c9", fg="#fff").pack(fill="x", padx=14, pady=3)
        FlatButton(parent, text="💾  Save Result",
                   command=self._save_result,
                   colour=WARN, fg="#000").pack(fill="x", padx=14, pady=3)

    # ── Image canvas ─────────────────────────────────────────────────

    def _build_canvas(self, parent):
        # Step navigator
        nav = tk.Frame(parent, bg=BG)
        nav.pack(fill="x", pady=(0, 6))

        self.step_btns = []
        for i, lbl in enumerate(self.STEP_LABELS):
            b = tk.Button(
                nav, text=lbl, font=("Segoe UI", 8),
                bg=BORDER, fg=MUTED,
                activebackground=ACCENT, activeforeground="#000",
                relief="flat", bd=0, padx=8, pady=5, cursor="hand2",
                command=lambda i=i: self._show_step(i)
            )
            b.pack(side="left", padx=2)
            self.step_btns.append(b)

        # Canvas
        self.canvas = tk.Label(parent, bg="#08090c",
                               text="Open an image to begin",
                               font=FONT_TITL, fg=MUTED,
                               relief="flat")
        self.canvas.pack(fill="both", expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(parent, textvariable=self.status_var,
                 font=FONT_MONO, bg=BG, fg=MUTED,
                 anchor="w").pack(fill="x", pady=(4, 0))

    # ── Log panel ────────────────────────────────────────────────────

    def _build_log(self, parent):
        tk.Label(parent, text="PIPELINE LOG", font=FONT_HEAD,
                 bg=PANEL, fg=TEXT).pack(pady=(14, 4), padx=14, anchor="w")
        self.log = LogPanel(parent)
        self.log.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        FlatButton(parent, text="Clear Log",
                   command=self.log.clear,
                   colour=BORDER, fg=MUTED).pack(padx=14, pady=6, fill="x")

    # ═════════════════════════════════════════════════════════════════
    # LOGIC
    # ═════════════════════════════════════════════════════════════════

    def _toggle_otsu(self):
        if self.v_otsu.get():
            self.sl_thresh.config(state="disabled")
        else:
            self.sl_thresh.config(state="normal")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select microscope image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        self.image_path = path
        self.lbl_file.config(text=os.path.basename(path))
        self.log.log(f"[OPEN] {path}", ACCENT2)
        # Preview original immediately
        img = cv2.imread(path)
        if img is not None:
            self._display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def _build_config(self) -> PipelineConfig:
        k = self.v_blur.get()
        if k % 2 == 0:
            k += 1
        return PipelineConfig(
            blur_ksize       = k,
            threshold_value  = self.v_threshold.get(),
            use_otsu         = self.v_otsu.get(),
            morph_kernel_size= self.v_morph_k.get(),
            morph_iterations = self.v_morph_iter.get(),
            min_cell_area    = self.v_min_area.get(),
            max_cell_area    = self.v_max_area.get(),
            min_circularity  = self.v_min_circ.get(),
            use_watershed    = self.v_watershed.get(),
            dist_threshold   = self.v_dist_thr.get(),
        )

    def _run_pipeline(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please open an image first.")
            return
        self._set_busy(True)
        threading.Thread(target=self._pipeline_worker, daemon=True).start()

    def _pipeline_worker(self):
        try:
            cfg = self._build_config()
            det = RBCDetector(cfg)
            self.after(0, lambda: self.log.log("[RUN] Pipeline started…", ACCENT))

            r = det.run(self.image_path)
            self.result = r

            self.after(0, self._on_pipeline_done, r, cfg)
        except Exception as exc:
            self.after(0, lambda: self.log.log(f"[ERROR] {exc}", ERROR))
            self.after(0, lambda: self._set_busy(False))

    def _on_pipeline_done(self, r: PipelineResult, cfg: PipelineConfig):
        self.log.log(f"[DONE] Cells detected : {r.cell_count}", ACCENT)
        self.log.log(f"       Elapsed         : {r.elapsed_ms:.1f} ms", TEXT)
        self.log.log(f"       Watershed        : {'ON' if cfg.use_watershed else 'OFF'}", TEXT)
        self.log.log(f"       Otsu             : {'ON' if cfg.use_otsu else 'OFF'}", TEXT)
        self.log.log(f"       Blur kernel      : {cfg.blur_ksize}", TEXT)
        self.log.log(f"       Morph kernel     : {cfg.morph_kernel_size}×{cfg.morph_kernel_size}", TEXT)
        self.log.log(f"       Area filter      : {cfg.min_cell_area}–{cfg.max_cell_area} px²", TEXT)
        self.log.log(f"       Circularity ≥    : {cfg.min_circularity:.2f}", TEXT)
        self.log.log("─" * 38, MUTED)

        self._show_step(5)          # jump to final result
        self._highlight_step(5)
        self.status_var.set(
            f"✔  {r.cell_count} cells  |  {r.elapsed_ms:.0f} ms  "
            f"|  {'Watershed ON' if cfg.use_watershed else 'Watershed OFF'}"
        )
        self._set_busy(False)

    def _run_compare(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please open an image first.")
            return
        self._set_busy(True)
        threading.Thread(target=self._compare_worker, daemon=True).start()

    def _compare_worker(self):
        try:
            cfg = self._build_config()
            det = RBCDetector(cfg)
            self.after(0, lambda: self.log.log("[COMPARE] Running ±watershed…", WARN))
            results = det.compare_watershed(self.image_path)
            self.compare_res = results
            self.after(0, self._on_compare_done, results)
        except Exception as exc:
            self.after(0, lambda: self.log.log(f"[ERROR] {exc}", ERROR))
            self.after(0, lambda: self._set_busy(False))

    def _on_compare_done(self, results: dict):
        r_off = results[False]
        r_on  = results[True]
        self.log.log("╔══ COMPARISON ═══════════════════════╗", WARN)
        self.log.log(f"  Without watershed : {r_off.cell_count} cells  ({r_off.elapsed_ms:.0f} ms)", TEXT)
        self.log.log(f"  With    watershed : {r_on.cell_count}  cells  ({r_on.elapsed_ms:.0f} ms)", TEXT)
        delta = r_on.cell_count - r_off.cell_count
        sign  = "+" if delta >= 0 else ""
        self.log.log(f"  Δ cells           : {sign}{delta}", ACCENT if delta > 0 else ERROR)
        self.log.log("╚══════════════════════════════════════╝", WARN)

        # Display side-by-side comparison stitched horizontally
        h  = max(r_off.final.shape[0], r_on.final.shape[0])
        pad_off = self._pad_to_height(r_off.final, h)
        pad_on  = self._pad_to_height(r_on.final,  h)

        divider = np.zeros((h, 6, 3), dtype=np.uint8)
        divider[:, :] = [0, 208, 132]           # green divider

        stitched = np.hstack([pad_off, divider, pad_on])
        cv2.putText(stitched, "WITHOUT Watershed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(stitched, "WITH Watershed",
                    (pad_off.shape[1] + 16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)

        self._display(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        self.status_var.set(
            f"Comparison — Without: {r_off.cell_count}  |  With: {r_on.cell_count}"
        )
        self._set_busy(False)

    @staticmethod
    def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h >= target_h:
            return img
        pad = np.zeros((target_h - h, w, 3), dtype=np.uint8)
        return np.vstack([img, pad])

    def _show_step(self, idx: int):
        self.current_step = idx
        if self.result is None:
            return
        step_name = self.STEPS[idx]
        img = getattr(self.result, step_name, None)
        if img is None:
            self.log.log(f"[WARN] Step '{step_name}' not computed yet.", WARN)
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3 and step_name != "original":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._display(img)
        self._highlight_step(idx)

    def _highlight_step(self, idx: int):
        for i, b in enumerate(self.step_btns):
            b.config(bg=ACCENT if i == idx else BORDER,
                     fg="#000" if i == idx else MUTED)

    def _display(self, rgb: np.ndarray):
        """Fit an RGB numpy image into the canvas label."""
        canvas_w = self.canvas.winfo_width()  or 900
        canvas_h = self.canvas.winfo_height() or 600
        h, w = rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self._photo_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self._photo_ref, text="")

    def _save_result(self):
        if self.result is None or self.result.final is None:
            messagebox.showwarning("Nothing to save", "Run the pipeline first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")]
        )
        if path:
            cv2.imwrite(path, self.result.final)
            self.log.log(f"[SAVE] {path}", ACCENT)
            messagebox.showinfo("Saved", f"Result saved to:\n{path}")

    def _set_busy(self, busy: bool):
        self.status_var.set("⏳  Processing…" if busy else self.status_var.get())
        cursor = "watch" if busy else ""
        self.config(cursor=cursor)


# ─────────────────────────────────────────────────────────────────────

def main():
    app = RBCApp()
    app.mainloop()


if __name__ == "__main__":
    main()
