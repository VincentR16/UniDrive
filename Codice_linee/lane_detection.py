"""
Algoritmo di Computer Vision per il riconoscimento dei limiti di corsia stradale.

Pipeline basata su morfologia top-hat:
    1. Grayscale + CLAHE
    2. Top-hat morfologico (isola strutture chiare sottili)
    3. Soglia di Otsu
    4. ROI geometrica
    5. Contorni filtrati per area e allungamento
    6. Fitting polinomiale

Espone due funzioni:
    - process_frame(frame)  -> dizionario con tutti gli stadi, usato da video
    - detect_lanes(path)    -> wrapper che legge da file
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MAX_LANES = 2

# ---------------------------------------------------------------------------
# Parametri configurabili
# ---------------------------------------------------------------------------

TOPHAT_KERNEL_SIZE = 30
ROI_TOP_RATIO      = 0.50
MIN_CONTOUR_AREA   = 500
MIN_ASPECT_RATIO   = 3.0
POLY_DEGREE        = 2

# Pre-calcoliamo i kernel una volta sola: molto piu' veloce in un loop video
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


def apply_roi(mask: np.ndarray, top_ratio: float = ROI_TOP_RATIO) -> np.ndarray:
    h, w = mask.shape[:2]
    roi = np.zeros_like(mask)
    cv2.rectangle(roi, (0, int(h * top_ratio)), (w, h), 255, -1)
    return cv2.bitwise_and(mask, roi)


#def find_lane_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    valid = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        (_, (w, h), _) = cv2.minAreaRect(cnt)
        if min(w, h) == 0:
            continue
        if max(w, h) / min(w, h) < MIN_ASPECT_RATIO:
            continue
        valid.append(cnt)
    return valid

def find_lane_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Contorni filtrati per area e forma, limitati ai MAX_LANES piu' grandi."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        (_, (w, h), _) = cv2.minAreaRect(cnt)
        if min(w, h) == 0:
            continue
        if max(w, h) / min(w, h) < MIN_ASPECT_RATIO:
            continue
        valid.append((area, cnt))

    # Ordina per area decrescente e tieni solo i primi MAX_LANES
    valid.sort(key=lambda x: x[0], reverse=True)
    return [cnt for _, cnt in valid[:MAX_LANES]]

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


def draw_results(image: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    output = image.copy()
    for i, cnt in enumerate(contours):
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        curve = fit_polynomial_curve(cnt)
        if curve is not None:
            cv2.polylines(output, [curve.reshape(-1, 1, 2)],
                          isClosed=False, color=(0, 0, 255), thickness=4)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, f"Lane {i + 1}", (cx - 30, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
    return output


# ---------------------------------------------------------------------------
# API pubbliche
# ---------------------------------------------------------------------------

def process_frame(frame: np.ndarray) -> dict:
    """
    Esegue la pipeline su un singolo frame gia' caricato in memoria.
    Questa e' la funzione che chiamerai dal loop video.
    """
    gray     = preprocess(frame)
    tophat   = tophat_transform(gray)
    mask     = threshold_tophat(tophat)
    roi_mask = apply_roi(mask)
    contours = find_lane_contours(roi_mask)
    result   = draw_results(frame, contours)

    return {
        "original": frame,
        "gray":     gray,
        "tophat":   tophat,
        "mask":     mask,
        "roi":      roi_mask,
        "contours": contours,
        "result":   result,
    }


def detect_lanes(image_path: str | Path) -> dict:
    """Wrapper per l'uso su un singolo file immagine."""
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
