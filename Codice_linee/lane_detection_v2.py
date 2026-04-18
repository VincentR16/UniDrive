"""
Algoritmo di Computer Vision per il riconoscimento dei limiti di corsia stradale.

Pipeline:
    1. Grayscale + CLAHE
    2. Top-hat morfologico (isola strutture chiare sottili)
    3. Filtro colore BIANCO (luminanza alta + saturazione bassa)
    4. Combinazione delle maschere (AND)
    5. ROI geometrica
    6. Ricerca della miglior linea a SINISTRA e della miglior linea a DESTRA
    7. Fitting polinomiale
    8. Calcolo dello scostamento del veicolo dal centro della corsia
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Parametri configurabili
# ---------------------------------------------------------------------------

TOPHAT_KERNEL_SIZE = 30
ROI_TOP_RATIO      = 0.55
MIN_CONTOUR_AREA   = 500
MIN_ASPECT_RATIO   = 3.0
POLY_DEGREE        = 2

# Filtro colore BIANCO
WHITE_L_MIN = 200   # luminanza minima (canale L di LAB, 0-255)
WHITE_S_MAX = 50    # saturazione massima (canale S di HSV, 0-255)

# Soglia di "centrato": se |offset_ratio| < questo valore il veicolo e' centrato
CENTERED_THRESHOLD = 0.15

# Pre-calcolo kernel e CLAHE (ottimizzazione per il loop video)
_TOPHAT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE))
_CLEAN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLAHE         = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# Stadi della pipeline
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)


def tophat_transform(gray: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _TOPHAT_KERNEL)


def threshold_tophat(tophat: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(tophat, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _CLEAN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLEAN_KERNEL, iterations=2)
    return mask


def white_color_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Maschera dei pixel di colore bianco.
    Definizione: luminanza alta (L di LAB) + saturazione bassa (S di HSV).
    Esclude automaticamente giallo, rosso, blu e altre linee colorate.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    l_chan = lab[:, :, 0]
    s_chan = hsv[:, :, 1]

    bright      = l_chan >= WHITE_L_MIN
    unsaturated = s_chan <= WHITE_S_MAX
    return (bright & unsaturated).astype(np.uint8) * 255


def apply_roi(mask: np.ndarray, top_ratio: float = ROI_TOP_RATIO) -> np.ndarray:
    h, w = mask.shape[:2]
    roi = np.zeros_like(mask)
    cv2.rectangle(roi, (0, int(h * top_ratio)), (w, h), 255, -1)
    return cv2.bitwise_and(mask, roi)


def find_left_right_lanes(mask: np.ndarray) -> tuple:
    """
    Trova la migliore linea a SINISTRA e la migliore linea a DESTRA.
    La classificazione si basa sulla posizione del centroide di ogni contorno
    rispetto al centro dell'immagine.

    Ritorna (contour_left, contour_right): ciascuno puo' essere None
    se non ci sono linee valide in quella meta'.
    """
    h, w = mask.shape[:2]
    image_cx = w // 2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    best_left  = (0, None)   # (area, contour)
    best_right = (0, None)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        (_, (cw, ch), _) = cv2.minAreaRect(cnt)
        if min(cw, ch) == 0:
            continue
        if max(cw, ch) / min(cw, ch) < MIN_ASPECT_RATIO:
            continue

        # Classifica per centroide X
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cnt_cx = M["m10"] / M["m00"]

        if cnt_cx < image_cx:
            if area > best_left[0]:
                best_left = (area, cnt)
        else:
            if area > best_right[0]:
                best_right = (area, cnt)

    return best_left[1], best_right[1]


def compute_center_offset(left_cnt, right_cnt, image_width: int,
                          image_height: int):
    """
    Scostamento del veicolo rispetto al centro della corsia.
    Usa i punti bassi dei contorni (piu' vicini al veicolo).

    Convenzione:
      offset_px > 0  -> veicolo spostato a DESTRA (deve sterzare a sinistra)
      offset_px < 0  -> veicolo spostato a SINISTRA (deve sterzare a destra)
    """
    if left_cnt is None or right_cnt is None:
        return None

    def bottom_x(cnt):
        pts = cnt.reshape(-1, 2)
        y_thresh = np.percentile(pts[:, 1], 75)
        bottom_pts = pts[pts[:, 1] >= y_thresh]
        return float(np.mean(bottom_pts[:, 0]))

    left_x  = bottom_x(left_cnt)
    right_x = bottom_x(right_cnt)

    lane_center  = (left_x + right_x) / 2
    image_center = image_width / 2

    offset_px       = image_center - lane_center
    lane_half_width = (right_x - left_x) / 2
    offset_ratio    = offset_px / lane_half_width if lane_half_width > 0 else 0.0

    return {
        "offset_px":      offset_px,
        "offset_ratio":   offset_ratio,
        "lane_center_x":  lane_center,
        "image_center_x": image_center,
        "left_x":         left_x,
        "right_x":        right_x,
    }


def fit_polynomial_curve(contour: np.ndarray, degree: int = POLY_DEGREE):
    pts = contour.reshape(-1, 2)
    xs  = pts[:, 0].astype(np.float32)
    ys  = pts[:, 1].astype(np.float32)

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()

    if x_range >= y_range:
        if len(np.unique(xs)) < degree + 1:
            return None
        coeffs = np.polyfit(xs, ys, degree)
        x_plot = np.linspace(xs.min(), xs.max(), num=100)
        y_plot = np.polyval(coeffs, x_plot)
    else:
        if len(np.unique(ys)) < degree + 1:
            return None
        coeffs = np.polyfit(ys, xs, degree)
        y_plot = np.linspace(ys.min(), ys.max(), num=100)
        x_plot = np.polyval(coeffs, y_plot)

    return np.column_stack((x_plot, y_plot)).astype(np.int32)


def draw_results(image: np.ndarray, left_cnt, right_cnt, offset_info) -> np.ndarray:
    """Sinistra=ciano, destra=arancio, HUD di centraggio in basso."""
    output = image.copy()
    h, w = output.shape[:2]

    # --- Linea SINISTRA (ciano) ---
    if left_cnt is not None:
        cv2.drawContours(output, [left_cnt], -1, (255, 200, 0), 2)
        curve = fit_polynomial_curve(left_cnt)
        if curve is not None:
            cv2.polylines(output, [curve.reshape(-1, 1, 2)],
                          isClosed=False, color=(255, 200, 0), thickness=4)
        M = cv2.moments(left_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, "LEFT", (cx - 30, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

    # --- Linea DESTRA (arancio) ---
    if right_cnt is not None:
        cv2.drawContours(output, [right_cnt], -1, (0, 140, 255), 2)
        curve = fit_polynomial_curve(right_cnt)
        if curve is not None:
            cv2.polylines(output, [curve.reshape(-1, 1, 2)],
                          isClosed=False, color=(0, 140, 255), thickness=4)
        M = cv2.moments(right_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, "RIGHT", (cx - 40, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

    # --- Indicatore di centraggio ---
    if offset_info is not None:
        img_cx  = int(offset_info["image_center_x"])
        lane_cx = int(offset_info["lane_center_x"])

        # Linea verde: dove DOVREBBE essere il centro (asse ottico)
        cv2.line(output, (img_cx, h - 90), (img_cx, h - 20), (0, 255, 0), 2)
        # Linea gialla: dove E' il centro corsia
        cv2.line(output, (lane_cx, h - 90), (lane_cx, h - 20), (0, 255, 255), 3)
        if abs(img_cx - lane_cx) > 3:
            cv2.arrowedLine(output, (img_cx, h - 55), (lane_cx, h - 55),
                            (255, 255, 255), 2, tipLength=0.3)

        offset_px    = offset_info["offset_px"]
        offset_ratio = offset_info["offset_ratio"]

        if abs(offset_ratio) < CENTERED_THRESHOLD:
            status = "CENTRATO"
            color  = (0, 255, 0)
        elif offset_ratio > 0:
            status = f"DRIFT DX  {abs(offset_px):.0f}px"
            color  = (0, 165, 255)
        else:
            status = f"DRIFT SX  {abs(offset_px):.0f}px"
            color  = (0, 165, 255)

        cv2.rectangle(output, (10, h - 45), (360, h - 8), (0, 0, 0), -1)
        cv2.putText(output, status, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(output, (10, h - 45), (400, h - 8), (0, 0, 0), -1)
        cv2.putText(output, "CORSIA NON RILEVATA", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return output


# ---------------------------------------------------------------------------
# API pubbliche
# ---------------------------------------------------------------------------

def process_frame(frame: np.ndarray) -> dict:
    """Esegue la pipeline completa su un singolo frame."""
    gray        = preprocess(frame)
    tophat      = tophat_transform(gray)
    mask_tophat = threshold_tophat(tophat)

    # Filtro colore bianco + combinazione con il top-hat
    mask_white  = white_color_mask(frame)
    mask        = cv2.bitwise_and(mask_tophat, mask_white)

    roi_mask    = apply_roi(mask)

    left_cnt, right_cnt = find_left_right_lanes(roi_mask)
    offset_info = compute_center_offset(left_cnt, right_cnt,
                                        frame.shape[1], frame.shape[0])
    result      = draw_results(frame, left_cnt, right_cnt, offset_info)

    # Lista dei contorni presenti (compatibilita' con video_lane_detection.py)
    contours = [c for c in (left_cnt, right_cnt) if c is not None]

    return {
        "original":     frame,
        "gray":         gray,
        "tophat":       tophat,
        "mask_white":   mask_white,
        "mask":         mask,
        "roi":          roi_mask,
        "left":         left_cnt,
        "right":        right_cnt,
        "offset_info":  offset_info,
        "contours":     contours,
        "result":       result,
    }


def detect_lanes(image_path: str | Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Impossibile caricare {image_path}")
    return process_frame(image)


# ---------------------------------------------------------------------------
# Entry point CLI (singola immagine)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python lane_detection.py <input_image> [output_image]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "lane_output.png"

    data = detect_lanes(input_path)
    cv2.imwrite(output_path, data["result"])

    print(f"Linee rilevate: {len(data['contours'])}  ->  {output_path}")
    if data["offset_info"] is not None:
        info = data["offset_info"]
        print(f"Offset: {info['offset_px']:+.1f} px  "
              f"(ratio {info['offset_ratio']:+.2f})")
