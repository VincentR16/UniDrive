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
ROI_TOP_RATIO      = 0.80
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



def roi_left(w, h, mask):
    left_roi = np.zeros_like(mask)
    left_poly = np.array([[
        (0, h),
        (int(w * 0.05), int(h * 0.65)),
        (int(w * 0.45), int(h * 0.65)),
        (int(w * 0.48), h)
    ]], dtype=np.int32)
    cv2.fillPoly(left_roi, left_poly, 255)
    left_mask = cv2.bitwise_and(mask, left_roi)
    return left_mask

def roi_right(w, h, mask): 
    right_roi = np.zeros_like(mask)
    right_poly = np.array([[
        (int(w * 0.52), h),
        (int(w * 0.55), int(h * 0.65)),
        (int(w * 0.95), int(h * 0.65)),
        (w, h)
    ]], dtype=np.int32)
    cv2.fillPoly(right_roi, right_poly, 255)
    right_mask = cv2.bitwise_and(mask, right_roi)
    return right_mask


def find_left_right_lanes(mask: np.ndarray) -> tuple:
    """
    Cerca la corsia sinistra e destra in due ROI distinte.
    Impone un vincolo duro:
    - LEFT deve stare nella metà sinistra dello schermo
    - RIGHT deve stare nella metà destra dello schermo
    """

    h, w = mask.shape[:2]
    image_cx = w / 2.0

    left_mask = roi_left(w, h, mask)
    right_mask = roi_right(w, h, mask)

    def contour_bottom_x(cnt):
        pts = cnt.reshape(-1, 2)
        y_thresh = np.percentile(pts[:, 1], 75)
        bottom_pts = pts[pts[:, 1] >= y_thresh]
        if len(bottom_pts) == 0:
            return None
        return float(np.mean(bottom_pts[:, 0]))

    def select_best_contour(side_mask, side="left"):
        contours, _ = cv2.findContours(
            side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        best_score = -1.0
        best_cnt = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            (_, (cw, ch), _) = cv2.minAreaRect(cnt)
            if min(cw, ch) == 0:
                continue

            aspect_ratio = max(cw, ch) / min(cw, ch)
            if aspect_ratio < MIN_ASPECT_RATIO:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            verticality = bh / max(bw, 1)

            bx = contour_bottom_x(cnt)
            if bx is None:
                continue

            # ---- VINCOLO DURO DI LATO ----
            if side == "left":
                # LEFT deve stare davvero nella metà sinistra
                if bx >= image_cx:
                    continue
                # opzionale: scarta anche contorni troppo centrali/destri
                if bx > w * 0.48:
                    continue
                side_penalty = abs(bx - w * 0.25)

            else:
                # RIGHT deve stare davvero nella metà destra
                if bx <= image_cx:
                    continue
                # opzionale: scarta anche contorni troppo centrali/sinistri
                if bx < w * 0.52:
                    continue
                side_penalty = abs(bx - w * 0.75)

            score = (
                area * 1.0 +
                aspect_ratio * 180.0 +
                verticality * 120.0 -
                side_penalty * 0.8
            )

            if score > best_score:
                best_score = score
                best_cnt = cnt

        return best_cnt

    left_cnt = select_best_contour(left_mask, side="left")
    right_cnt = select_best_contour(right_mask, side="right")

    # ---- CONTROLLO FINALE DURO ----
    if left_cnt is not None:
        lx = contour_bottom_x(left_cnt)
        if lx is None or lx >= image_cx:
            left_cnt = None

    if right_cnt is not None:
        rx = contour_bottom_x(right_cnt)
        if rx is None or rx <= image_cx:
            right_cnt = None

    # Se entrambe esistono, devono essere ordinate correttamente
    if left_cnt is not None and right_cnt is not None:
        lx = contour_bottom_x(left_cnt)
        rx = contour_bottom_x(right_cnt)

        if lx is None or rx is None:
            return None, None

        # Vieta inversioni o linee troppo vicine
        if lx >= rx or abs(rx - lx) < w * 0.15:
            return None, None

    return left_cnt, right_cnt

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
        cv2.drawContours(output, [left_cnt], -1, (0, 140, 255), 2)
        curve = fit_polynomial_curve(left_cnt)
        if curve is not None:
            cv2.polylines(output, [curve.reshape(-1, 1, 2)],
                          isClosed=False, color=(0, 140, 255), thickness=4)


    # --- Linea DESTRA (arancio) ---
    if right_cnt is not None:
        cv2.drawContours(output, [right_cnt], -1, (0, 140, 255), 2)
        curve = fit_polynomial_curve(right_cnt)
        if curve is not None:
            cv2.polylines(output, [curve.reshape(-1, 1, 2)],
                          isClosed=False, color=(0, 140, 255), thickness=4)

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
