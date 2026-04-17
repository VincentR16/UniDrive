"""
Esecuzione del rilevamento di corsia su video (webcam, file, o stream RTSP).

Esempi d'uso:
    # Webcam default (indice 0)
    python video_lane_detection.py 0

    # File video
    python video_lane_detection.py mio_video.mp4

    # Stream IP / RTSP (tipico per robot o telecamera di rete)
    python video_lane_detection.py rtsp://192.168.1.50:554/stream

    # Salva anche il video annotato su disco
    python video_lane_detection.py mio_video.mp4 --save output.mp4

    # Mostra pannello di debug (4 pannelli) invece del solo risultato
    python video_lane_detection.py 0 --debug

Controlli da tastiera durante l'esecuzione:
    q / ESC  -> esci
    SPACE    -> pausa / riprendi
    s        -> salva uno screenshot del frame corrente
    d        -> attiva/disattiva vista debug
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lane_detection import process_frame


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def parse_source(src: str):
    """Accetta un intero (webcam), un percorso file, o una URL."""
    if src.isdigit():
        return int(src)
    return src


def build_debug_view(data: dict, size=(640, 360)) -> np.ndarray:
    """Pannello 2x2 compatto per vedere gli stadi durante il video."""
    def label(img, text, is_gray=False):
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, size)
        cv2.rectangle(img, (0, 0), (size[0], 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    row1 = np.hstack([
        label(data["original"], "Originale"),
        label(data["tophat"],   "Top-hat", is_gray=True),
    ])
    row2 = np.hstack([
        label(data["roi"],      "Maschera + ROI", is_gray=True),
        label(data["result"],   "Risultato"),
    ])
    return np.vstack([row1, row2])


def draw_hud(frame: np.ndarray, fps: float, n_lanes: int) -> None:
    """Sovrascrive in-place un piccolo HUD in alto a sinistra."""
    txt = f"FPS: {fps:5.1f}   Lanes: {n_lanes}"
    cv2.rectangle(frame, (5, 5), (260, 35), (0, 0, 0), -1)
    cv2.putText(frame, txt, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(source, save_path: str | None = None, debug: bool = False) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire la sorgente '{source}'")
        sys.exit(1)

    # Proprieta' del video per il writer
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_in, (width, height))
        print(f"Salvataggio su: {save_path}")

    window_name = "Lane Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused         = False
    show_debug     = debug
    last_time      = time.time()
    fps_smooth     = 0.0
    screenshot_idx = 0

    print("Controlli: [q/ESC]=esci  [SPACE]=pausa  [s]=screenshot  [d]=debug")

    total_lanes = 0
    total_frames = 0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Fine del video o errore di lettura.")
                break

            # Pipeline principale
            data = process_frame(frame)
            total_lanes += len(data["contours"])
            total_frames += 1
            output = data["result"]

            # Scrittura su file (solo vista 'result', non debug)
            if writer is not None:
                writer.write(output)

            # FPS con smoothing esponenziale
            now = time.time()
            dt  = now - last_time
            last_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps_smooth  = 0.9 * fps_smooth + 0.1 * instant_fps

            draw_hud(output, fps_smooth, len(data["contours"]))
            display = build_debug_view(data) if show_debug else output

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # q o ESC
            break
        elif key == ord(" "):              # pausa
            paused = not paused
        elif key == ord("d"):              # toggle debug
            show_debug = not show_debug
        elif key == ord("s"):              # screenshot
            fn = f"screenshot_{screenshot_idx:03d}.png"
            cv2.imwrite(fn, display)
            print(f"Screenshot salvato: {fn}")
            screenshot_idx += 1

    print(f"\n--- Statistiche ---")
    print(f"Frame processati:       {total_frames}")
    print(f"Linee totali rilevate:  {total_lanes}")
    if total_frames > 0:
        print(f"Media linee per frame:  {total_lanes / total_frames:.2f}")
        
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane detection su video.")
    parser.add_argument("source",
                        help="Indice webcam (es. 0), percorso file, o URL RTSP")
    parser.add_argument("--save", default=None,
                        help="Percorso output .mp4 per salvare il video annotato")
    parser.add_argument("--debug", action="store_true",
                        help="Mostra il pannello 2x2 con gli stadi intermedi")
    args = parser.parse_args()

    run(parse_source(args.source), save_path=args.save, debug=args.debug)
