"""
Pipeline overview:
  1. Load image
  2. Resize (normalisation géométrique)        
  3. ROI masking (région d'intérêt)            
  4. Convert to grayscale
  5. Gaussian blur  → noise reduction
  6. Canny edge detection (visualisation)      
  7. Thresholding   → binary mask
  8. Morphological ops (erosion / dilation / opening / closing)
  9. Distance transform + Watershed → separate touching cells
  10. Contour detection + shape filtering
  11. Count & visualise
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
    # ── Transformation géométrique ──────────────────────────────────
    target_width: int        = 800    # largeur cible pour le redimensionnement
    target_height: int       = 600    # hauteur cible pour le redimensionnement

    # ── ROI (Region of Interest) ────────────────────────────────────
    use_roi: bool            = False  # activer le masque ROI
    roi_x: int               = 0      # coin supérieur gauche X
    roi_y: int               = 0      # coin supérieur gauche Y
    roi_w: int               = 400    # largeur de la ROI
    roi_h: int               = 400    # hauteur de la ROI

    # ── Flou gaussien ───────────────────────────────────────────────
    blur_ksize: int          = 7      # Gaussian kernel (must be odd)

    # ── Détection de contours Canny ─────────────────────────────────
    canny_low: int           = 30     # seuil bas Canny
    canny_high: int          = 120    # seuil haut Canny

    # ── Seuillage ───────────────────────────────────────────────────
    threshold_value: int     = 128    # manual threshold (0–255)
    use_otsu: bool           = True   # override threshold_value with Otsu

    # ── Morphologie ─────────────────────────────────────────────────
    morph_kernel_size: int   = 3      # structuring element size
    morph_iterations: int    = 2      # erosion / dilation iterations

    # ── Filtres de forme ────────────────────────────────────────────
    min_cell_area: int       = 400    # px² — discard tiny blobs
    max_cell_area: int       = 8000   # px² — discard huge artefacts
    min_circularity: float   = 0.55   # 1.0 = perfect circle

    # ── Watershed ───────────────────────────────────────────────────
    use_watershed: bool      = True   # toggle watershed separation
    dist_threshold: float    = 0.5    # fraction of max for sure-fg


@dataclass
class PipelineResult:
    """Contient toutes les images intermédiaires produites par le pipeline."""
    original:      Optional[np.ndarray] = None
    resized:       Optional[np.ndarray] = None   
    roi_mask:      Optional[np.ndarray] = None   
    roi_applied:   Optional[np.ndarray] = None   
    gray:          Optional[np.ndarray] = None
    blurred:       Optional[np.ndarray] = None
    edges:         Optional[np.ndarray] = None   
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
    Encapsule le pipeline complet de vision par ordinateur classique
    pour la détection des globules rouges (RBC).

    Chaque méthode correspond à une étape logique du traitement
    et peut être appelée indépendamment pour une utilisation
    interactive ou via une interface graphique (GUI).
    """

    def __init__(self, config: PipelineConfig = None):
        self.cfg = config or PipelineConfig()
        self.result = PipelineResult()

    # ── Step 1: Load ────────────────────────────────────────────────

    def load_image(self, path: str) -> np.ndarray:
        """
        Lit une image à partir du disque.
        OpenCV charge en BGR par défaut; nous conservons cette convention tout au long
        et convertissons en RGB uniquement lorsqu'on transmet à Matplotlib/Tkinter.
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        self.result.original = img.copy()
        return img

    # ── Step 2: Resize (Transformation géométrique) ──────────────────

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Redimensionne l'image à une taille cible normalisée.

        Pourquoi redimensionner ?
        - Normalise les entrées quelle que soit la résolution du microscope.
        - Réduit le temps de calcul sur les images haute résolution.
        - Garantit que les paramètres de filtre (kernel, aire) restent cohérents.

        cv2.INTER_AREA est l'interpolation recommandée pour la réduction
        car elle évite les artefacts de moiré sur les structures fines.
        """
        resized = cv2.resize(
            img,
            (self.cfg.target_width, self.cfg.target_height),
            interpolation=cv2.INTER_AREA
        )
        self.result.resized = resized.copy()
        return resized

    # ── Step 3: ROI Masking (Region of Interest) ────────────────────

    def apply_roi(self, img: np.ndarray) -> np.ndarray:
        """
        Applique un masque de région d'intérêt (ROI) sur l'image.

        Technique : cv2.bitwise_and avec un masque binaire.

        Avantages :
        - Concentre l'analyse sur la zone utile (ex: lame sans bordure).
        - Évite que les artefacts de bord du microscope (vignettage, texte)
          ne perturbent le seuillage et la détection.

        Si use_roi est désactivé, l'image est renvoyée telle quelle
        avec un masque plein pour la cohérence du pipeline.
        """
        h, w = img.shape[:2]

        if $not self.cfg.use_roi:
            # Masque plein (toute l'image)
            mask = np.ones((h, w), dtype=np.uint8) * 255
            self.result.roi_mask    = mask
            self.result.roi_applied = img.copy()
            return img

        # Construire le masque rectangulaire ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y = self.cfg.roi_x, self.cfg.roi_y
        rw   = min(self.cfg.roi_w, w - x)
        rh   = min(self.cfg.roi_h, h - y)
        mask[y:y+rh, x:x+rw] = 255

        # Appliquer le masque avec bitwise_and
        roi_applied = cv2.bitwise_and(img, img, mask=mask)

        self.result.roi_mask    = mask
        self.result.roi_applied = roi_applied
        return roi_applied

    # ── Step 4: Grayscale ────────────────────────────────────────────

    def to_gray(self, img: np.ndarray) -> np.ndarray:
        """
        Convertit une image BGR en niveaux de gris.
        Les globules rouges (RBC) apparaissent comme des disques plus sombres
        ou plus clairs selon la coloration utilisée ;
        les niveaux de gris conservent les informations de luminance
        nécessaires pour le seuillage.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.result.gray = gray
        return gray

    # ── Step 5: Noise reduction ──────────────────────────────────────

    def blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Le flou gaussien réduit le bruit de haute fréquence qui pourrait
        sinon produire de faux contours ou de fausses régions lors du seuillage.

        La taille du noyau (kernel) doit être impaire ;
        plus elle est grande, plus le lissage est fort.
        """
        k = self.cfg.blur_ksize
        if k % 2 == 0:
            k += 1                            # enforce oddness
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        self.result.blurred = blurred
        return blurred

    # ── Step 6: Canny Edge Detection ────────────────────────────────

    def detect_edges_canny(self, blurred: np.ndarray) -> np.ndarray:
        """
        Détecte les contours avec l'algorithme de Canny.

        Canny applique en interne :
          1. Calcul du gradient (Sobel X et Y) pour trouver l'intensité des bords.
          2. Suppression des non-maxima → amincissement des bords.
          3. Double seuillage (low / high) :
             - Pixels > high  : bords certains (forts)
             - Pixels < low   : rejetés
             - Entre low/high : conservés seulement s'ils touchent un bord fort.

        Rôle dans ce pipeline :
        - Étape de VISUALISATION permettant d'inspecter la qualité
          du flou gaussien et d'évaluer les contours réels des cellules.
        - Non utilisée directement pour la détection finale (remplacée
          par le seuillage + watershed), mais très utile pour le réglage
          des paramètres et la démonstration pédagogique.

        Paramètres clés (configurables via l'interface) :
          canny_low  : seuil bas  — plus bas = plus de bords détectés
          canny_high : seuil haut — plus haut = seulement les bords forts
        """
        edges = cv2.Canny(
            blurred,
            self.cfg.canny_low,
            self.cfg.canny_high
        )
        self.result.edges = edges
        return edges

    # ── Step 7: Thresholding ─────────────────────────────────────────

    def threshold(self, blurred: np.ndarray) -> np.ndarray:
        """
        Convertit une image en niveaux de gris en masque binaire.

        Méthode d'Otsu (par défaut) :
        Choisit automatiquement le seuil qui minimise la variance
        intra-classe — très efficace pour les histogrammes bimodaux
        (cellules vs arrière-plan).

        Seuil manuel :
        Utile lorsque l'éclairage est très uniforme et que l'utilisateur
        souhaite un contrôle précis via le curseur de l'interface graphique (GUI).
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

    # ── Step 8: Morphological operations ─────────────────────────────

    def morphology(self, thresh: np.ndarray) -> np.ndarray:
        """
        Quatre opérations morphologiques classiques appliquées en séquence :

        EROSION   — réduit les régions blanches ; supprime les fines protrusions
                    et les pixels de bruit ayant survécu au seuillage.

        DILATATION — agrandit les régions blanches ; comble les petits espaces
                    à l'intérieur des cellules.

        OUVERTURE = érosion puis dilatation → supprime les petits points lumineux
                    (bruit) tout en préservant la forme des cellules.

        FERMETURE = dilatation puis érosion → remplit les petits trous sombres
                    à l'intérieur des cellules causés par des artefacts de coloration.
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

    # ── Step 9a: Distance Transform ──────────────────────────────────

    def distance_transform(self, morphed: np.ndarray) -> np.ndarray:
        """
        Pour chaque pixel du premier plan, calcule sa distance euclidienne
        par rapport au pixel d'arrière-plan le plus proche.

        Les pics de cette carte correspondent aux centres des cellules,
        car ils sont éloignés de tout contour.

        Cette transformation est utilisée par l'algorithme Watershed
        pour générer les marqueurs (premier plan certain).
        """
        dist = cv2.distanceTransform(morphed, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        self.result.distance = dist
        return dist

    # ── Step 9b: Watershed ───────────────────────────────────────────

    def watershed(
        self,
        original: np.ndarray,
        morphed:  np.ndarray,
        dist:     np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        L'algorithme Watershed permet de séparer des cellules collées
        ou qui se chevauchent.

        Idée principale — considérer la carte de distance inversée comme
        une surface topographique.
        L'« eau » remplit les bassins à partir des pics initiaux ;
        les lignes de partage des eaux se forment sur les crêtes entre
        les bassins → ces crêtes deviennent les frontières des cellules.

        Étapes :
        1. sure_bg  = masque morphologique dilaté (arrière-plan certain)
        2. sure_fg  = carte de distance seuillée (centres des cellules certains)
        3. unknown  = sure_bg - sure_fg (pixels frontières incertains)
        4. Étiqueter les composantes connexes dans sure_fg
        5. Marquer la région unknown avec 0 (laisser Watershed décider)
        6. Appliquer cv2.watershed → frontières marquées par -1
        """
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg  = cv2.dilate(morphed, kernel, iterations=3)

        _, sure_fg = cv2.threshold(
            dist, self.cfg.dist_threshold * dist.max(), 255, 0
        )
        sure_fg  = np.uint8(sure_fg)
        unknown  = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers   += 1
        markers[unknown == 255] = 0

        img_ws    = original.copy()
        markers   = cv2.watershed(img_ws, markers)
        img_ws[markers == -1] = [0, 0, 255]

        self.result.watershed_map = markers
        return markers, img_ws

    # ── Step 10: Contour detection & filtering ────────────────────────

    def detect_contours(
        self,
        markers:  Optional[np.ndarray],
        morphed:  np.ndarray,
        original: np.ndarray
    ) -> Tuple[np.ndarray, List, int]:
        """
        Détecte et filtre les contours représentant des globules rouges (RBC)
        individuels.

        Critères de filtrage (configurables) :
        • Aire         : élimine les poussières et les groupes de cellules
        • Circularité  : 4π·Aire / Périmètre² — les RBC sont ~circulaires
        """
        output = original.copy()
        valid_contours = []

        if self.cfg.use_watershed and markers is not None:
            for label in np.unique(markers):
                if label <= 1:
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
            cnts, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid_contours = [c for c in cnts if self._is_valid(c)]

        for i, c in enumerate(valid_contours, start=1):
            (x, y), r = cv2.minEnclosingCircle(c)
            cx, cy, cr = int(x), int(y), int(r)
            cv2.circle(output, (cx, cy), cr, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 3,  (0, 0, 255), -1)
            cv2.putText(
                output, str(i), (cx - 8, cy - cr - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                cv2.LINE_AA
            )

        count = len(valid_contours)
        label = f"RBC Count: {count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(output, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)
        cv2.putText(
            output, label, (14, 14 + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2, cv2.LINE_AA
        )

        self.result.final      = output
        self.result.contours   = valid_contours
        self.result.cell_count = count
        return output, valid_contours, count

    # ── Helper: shape filter ─────────────────────────────────────────

    def _is_valid(self, contour: np.ndarray) -> bool:
        """
        Rejette les contours qui ne correspondent pas à des RBC individuels.

        Circularité = 4π·A / P²
          = 1.0  → cercle parfait
          ≈ 0.75 → légèrement allongé
          < 0.5  → très irrégulier (bruit ou deux cellules fusionnées)
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
        Exécute le pipeline complet de bout en bout et retourne un objet
        PipelineResult contenant toutes les images intermédiaires ainsi que
        le comptage final.

        Ordre des étapes :
          1. Chargement
          2. Redimensionnement (resize)     ← NOUVEAU
          3. Masque ROI (bitwise_and)       ← NOUVEAU
          4. Niveaux de gris
          5. Flou gaussien
          6. Détection de bords Canny       ← NOUVEAU
          7. Seuillage (Otsu / manuel)
          8. Morphologie (érosion, dilatation, ouverture, fermeture)
          9. Transformée de distance + Watershed
          10. Détection et filtrage des contours
        """
        t0 = time.perf_counter()

        img      = self.load_image(image_path)
        img      = self.resize_image(img)          # étape 2 — resize
        img      = self.apply_roi(img)             # étape 3 — ROI masking
        gray     = self.to_gray(img)
        blurred  = self.blur(gray)
        _        = self.detect_edges_canny(blurred)  # étape 6 — Canny (stocké dans result.edges)
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
        Exécute le pipeline deux fois — avec et sans Watershed — et retourne
        les deux résultats afin de permettre une analyse côte à côte.
        """
        results = {}
        for ws in (False, True):
            self.cfg.use_watershed = ws
            self.result = PipelineResult()
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
    print(f"  Resize       : {cfg.target_width}x{cfg.target_height}")
    print(f"  ROI active   : {'YES' if cfg.use_roi else 'NO'}")
    print(f"  Cells found  : {r.cell_count}")
    print(f"  Elapsed      : {r.elapsed_ms:.1f} ms")
    print(f"  Watershed    : {'ON' if cfg.use_watershed else 'OFF'}")
    print("=" * 60)

    steps = [
        ("Original",       cv2.cvtColor(r.original,    cv2.COLOR_BGR2RGB)),
        ("Resized",        cv2.cvtColor(r.resized,     cv2.COLOR_BGR2RGB)),
        ("ROI Applied",    cv2.cvtColor(r.roi_applied, cv2.COLOR_BGR2RGB)),
        ("Grayscale",      r.gray),
        ("Blurred",        r.blurred),
        ("Canny Edges",    r.edges),
        ("Thresholded",    r.thresholded),
        ("Morphological",  r.morphed),
        ("Final Detection",cv2.cvtColor(r.final,       cv2.COLOR_BGR2RGB)),
    ]

    if plt:
        fig = plt.figure(figsize=(22, 10), facecolor="#0d0d0d")
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.15)
        fig.suptitle(
            f"RBC Detector — {r.cell_count} cells  |  {r.elapsed_ms:.0f} ms",
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