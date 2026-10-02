"""Rastreamento ocular por webcam para estudos de UX.

Fluxo de uma sessão (cada fase é exibida em tela cheia):

1. Pré-checagem   – verifica detecção do rosto e ajusta o limiar de piscada (EAR).
2. Calibração     – o participante olha para N alvos; treina-se uma regressão
                    características do olho/cabeça -> coordenadas da tela.
3. Validação      – alvos NOVOS (não usados no treino) medem acurácia e precisão.
4. Gravação       – o estímulo (ex.: print de um site) é exibido; o olhar é
                    estimado, suavizado e segmentado em fixações (I-DT).

Saídas em --output-dir: session.json, frames.csv, fixations.csv,
calibration_samples.csv (todas as amostras de calibração e validação, para
reanálise offline com analyze_session.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from gaze_core import (
    FEATURE_NAMES,
    GazeCalibrator,
    IDTFixationDetector,
    Letterbox,
    PointFilter,
    clean_calibration_samples,
    grid_points,
    px_per_degree,
    validation_metrics,
)

QUIT_KEYS = {ord("q"), 27}  # q ou ESC
SPACE = ord(" ")

# Índices do MediaPipe Face Mesh (refine_landmarks=True). O "olho A" usa os
# pontos 33/133 e a íris 468-472; o "olho B" usa 263/362 e a íris 473-477.
EYE_A = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": [468, 469, 470, 471, 472]}
EYE_B = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": [473, 474, 475, 476, 477]}
NOSE_TIP = 1

CALIBRATION_POINTS = {
    "5": [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)],
    "9": grid_points([0.1, 0.5, 0.9]),
    "13": grid_points([0.1, 0.5, 0.9]) + grid_points([0.3, 0.7]),
}
VALIDATION_POINTS = grid_points([0.3, 0.7]) + [(0.5, 0.5)]


# ---------------------------------------------------------------------------
# amostra produzida pelos backends
# ---------------------------------------------------------------------------
@dataclass
class GazeSample:
    eye_x: float          # posição da íris relativa ao centro do olho (÷ largura do olho)
    eye_y: float
    head_x: float         # posição do nariz relativa ao centro dos olhos (÷ distância interocular)
    head_y: float
    overlay_px: tuple[float, float]  # ponto para desenhar no frame da câmera
    blink: bool
    avg_ear: float

    @property
    def features(self) -> tuple[float, float, float, float]:
        return (self.eye_x, self.eye_y, self.head_x, self.head_y)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


class MediaPipeBackend:
    """Face Mesh com refinamento de íris (CNN, 478 pontos)."""

    name = "mediapipe"
    supports_blink = True

    def __init__(self, cv2_module) -> None:
        import mediapipe as mp

        self.cv2 = cv2_module
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.blink_threshold = 0.21

    def process(self, frame_bgr) -> Optional[GazeSample]:
        h, w = frame_bgr.shape[:2]
        result = self.mesh.process(self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark

        def px(i: int) -> tuple[float, float]:
            return lm[i].x * w, lm[i].y * h

        def eye(spec: dict):
            outer, inner = px(spec["outer"]), px(spec["inner"])
            width = _dist(outer, inner)
            if width < 1e-6:
                return None
            iris = [px(i) for i in spec["iris"]]
            cx = sum(p[0] for p in iris) / len(iris)
            cy = sum(p[1] for p in iris) / len(iris)
            mx, my = (outer[0] + inner[0]) / 2.0, (outer[1] + inner[1]) / 2.0
            ear = _dist(px(spec["top"]), px(spec["bottom"])) / width
            return (cx - mx) / width, (cy - my) / width, ear, (cx, cy)

        a, b = eye(EYE_A), eye(EYE_B)
        if a is None or b is None:
            return None
        a_outer, b_outer = px(EYE_A["outer"]), px(EYE_B["outer"])
        iod = _dist(a_outer, b_outer)
        if iod < 1e-6:
            return None
        mid = ((a_outer[0] + b_outer[0]) / 2.0, (a_outer[1] + b_outer[1]) / 2.0)
        nose = px(NOSE_TIP)
        avg_ear = (a[2] + b[2]) / 2.0
        return GazeSample(
            eye_x=(a[0] + b[0]) / 2.0,
            eye_y=(a[1] + b[1]) / 2.0,
            head_x=(nose[0] - mid[0]) / iod,
            head_y=(nose[1] - mid[1]) / iod,
            overlay_px=a[3],
            blink=avg_ear < self.blink_threshold,
            avg_ear=avg_ear,
        )

    def close(self) -> None:
        self.mesh.close()


class OpenCVBackend:
    """Fallback clássico: Haar cascades (Viola-Jones) + limiarização da pupila."""

    name = "opencv"
    supports_blink = False  # Haar não detecta olho fechado: piscada vira perda de dado

    def __init__(self, cv2_module) -> None:
        self.cv2 = cv2_module
        self.face = cv2_module.CascadeClassifier(cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eyes = cv2_module.CascadeClassifier(cv2_module.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
        self.blink_threshold = float("nan")

    @staticmethod
    def _is_reasonable_eye(ex, ey, ew, eh, roi_w, roi_h) -> bool:
        if ew <= 0 or eh <= 0:
            return False
        aspect = eh / max(ew, 1)
        cx, cy = ex + ew / 2.0, ey + eh / 2.0
        return 0.12 <= aspect <= 0.9 and cy <= roi_h * 0.72 and roi_w * 0.08 <= cx <= roi_w * 0.92

    def _pupil(self, gray_eye) -> tuple[float, float]:
        blur = self.cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, thresh = self.cv2.threshold(blur, 0, 255, self.cv2.THRESH_BINARY_INV + self.cv2.THRESH_OTSU)
        m = self.cv2.moments(thresh)
        hh, ww = gray_eye.shape[:2]
        if m["m00"] < 1:
            return 0.5, 0.5
        return m["m10"] / m["m00"] / max(ww, 1), m["m01"] / m["m00"] / max(hh, 1)

    def process(self, frame_bgr) -> Optional[GazeSample]:
        cv2 = self.cv2
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y:y + int(h * 0.6), x:x + w]
        eyes = [e for e in self.eyes.detectMultiScale(roi, 1.1, 4, minSize=(20, 20))
                if self._is_reasonable_eye(*e, w, roi.shape[0])]
        if not eyes:
            return None
        eyes = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]
        rx, ry, overlay = [], [], None
        for ex, ey, ew, eh in eyes:
            eye_roi = roi[ey:ey + eh, ex:ex + ew]
            if eye_roi.size == 0:
                continue
            px_, py_ = self._pupil(eye_roi)
            rx.append(px_ - 0.5)
            ry.append(py_ - 0.5)
            overlay = (x + ex + px_ * ew, y + ey + py_ * eh)
        if not rx:
            return None
        return GazeSample(
            eye_x=sum(rx) / len(rx),
            eye_y=sum(ry) / len(ry),
            head_x=(x + w / 2.0) / W - 0.5,
            head_y=(y + h / 2.0) / H - 0.5,
            overlay_px=overlay,
            blink=False,
            avg_ear=float("nan"),
        )

    def close(self) -> None:
        return None


def choose_backend(name: str, cv2_module):
    if name in {"auto", "mediapipe"}:
        try:
            return MediaPipeBackend(cv2_module)
        except Exception as exc:  # mediapipe ausente ou incompatível
            if name == "mediapipe":
                raise
            print(f"[aviso] MediaPipe indisponível ({exc}); usando OpenCV.")
    return OpenCVBackend(cv2_module)


# ---------------------------------------------------------------------------
# exibição
# ---------------------------------------------------------------------------
class CvDisplay:
    STIM_WIN = "Eye Tracking UX"
    CAM_WIN = "Eye Tracking UX - Camera"

    def __init__(self, cv2_module, width: int, height: int, show_camera: bool) -> None:
        self.cv2 = cv2_module
        self.width = width
        self.height = height
        self.show_camera_window = show_camera
        self.current_target: Optional[tuple[float, float]] = None

    def open(self) -> None:
        self.cv2.namedWindow(self.STIM_WIN, self.cv2.WINDOW_NORMAL)
        self.cv2.setWindowProperty(self.STIM_WIN, self.cv2.WND_PROP_FULLSCREEN, self.cv2.WINDOW_FULLSCREEN)

    def show(self, canvas, target: Optional[tuple[float, float]] = None) -> None:
        self.current_target = target
        self.cv2.imshow(self.STIM_WIN, canvas)

    def show_camera(self, frame) -> None:
        if self.show_camera_window:
            self.cv2.imshow(self.CAM_WIN, frame)

    def key(self) -> int:
        return self.cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        self.cv2.destroyAllWindows()


BG = (40, 40, 40)


def blank_canvas(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), BG, dtype=np.uint8)


def put_center(cv2, img, text: str, y: int, scale: float = 0.9, color=(235, 235, 235)) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_target(cv2, img, x: float, y: float, progress: float) -> None:
    """Alvo que encolhe ao longo do tempo (ajuda a manter o olhar no centro)."""
    r = int(22 - 12 * min(max(progress, 0.0), 1.0))
    cv2.circle(img, (int(x), int(y)), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), -1, cv2.LINE_AA)


def annotate_camera(cv2, frame, sample: Optional[GazeSample], label: str):
    out = frame.copy()
    cv2.putText(out, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if sample is None:
        cv2.putText(out, "rosto/olhos nao detectados", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.circle(out, (int(sample.overlay_px[0]), int(sample.overlay_px[1])), 4, (0, 255, 0), -1)
        if math.isfinite(sample.avg_ear):
            cv2.putText(out, f"EAR {sample.avg_ear:.3f} blink={sample.blink}", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------------
# sessão
# ---------------------------------------------------------------------------
class SessionAborted(Exception):
    pass


@dataclass
class Ctx:
    cv2: object
    cap: object
    backend: object
    display: object
    width: int
    height: int
    t0: float

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def read(self):
        for _ in range(200):
            ok, frame = self.cap.read()
            if ok:
                return frame
        raise RuntimeError("A webcam parou de enviar frames.")

    def pump(self, canvas, label: str, target=None) -> Optional[GazeSample]:
        """Lê um frame, processa, atualiza as janelas e verifica teclas."""
        frame = self.read()
        sample = self.backend.process(frame)
        self.display.show(canvas, target)
        self.display.show_camera(annotate_camera(self.cv2, frame, sample, label))
        if self.display.key() in QUIT_KEYS:
            raise SessionAborted()
        return sample


def wait_for_space(ctx: Ctx, lines: list[str]) -> None:
    """Tela de instrução; o participante aperta ESPAÇO quando estiver pronto."""
    while True:
        canvas = blank_canvas(ctx.width, ctx.height)
        y = ctx.height // 2 - 30 * len(lines)
        for line in lines:
            put_center(ctx.cv2, canvas, line, y)
            y += 50
        put_center(ctx.cv2, canvas, "[ESPACO] continuar    [Q] sair", y + 40, 0.7, (150, 200, 255))
        frame = ctx.read()
        sample = ctx.backend.process(frame)
        status = "rosto OK" if sample is not None else "ROSTO NAO DETECTADO - ajuste luz/posicao"
        put_center(ctx.cv2, canvas, status, ctx.height - 60, 0.7, (0, 220, 0) if sample else (0, 0, 255))
        ctx.display.show(canvas)
        ctx.display.show_camera(annotate_camera(ctx.cv2, frame, sample, "instrucoes"))
        k = ctx.display.key()
        if k in QUIT_KEYS:
            raise SessionAborted()
        if k == SPACE:
            return


def run_precheck(ctx: Ctx, seconds: float) -> dict:
    """Mede cobertura de detecção e calibra o limiar de piscada (EAR)."""
    wait_for_space(ctx, ["Pre-checagem", "Primeiro mantenha os olhos ABERTOS olhando a tela,",
                         "depois PISQUE naturalmente algumas vezes."])
    start = time.perf_counter()
    open_ears, blink_ears, seen, detected = [], [], 0, 0
    while (el := time.perf_counter() - start) < seconds:
        phase_open = el < seconds / 2
        canvas = blank_canvas(ctx.width, ctx.height)
        put_center(ctx.cv2, canvas, "Olhos abertos" if phase_open else "Pisque naturalmente", ctx.height // 2)
        sample = ctx.pump(canvas, "pre-checagem")
        seen += 1
        if sample is not None:
            detected += 1
            if math.isfinite(sample.avg_ear):
                (open_ears if phase_open else blink_ears).append(sample.avg_ear)

    result = {"frames": seen, "coverage": detected / seen if seen else 0.0,
              "blink_threshold": ctx.backend.blink_threshold}
    if ctx.backend.supports_blink and open_ears:
        # Mesmo na fase "pisque" o olho fica aberto >90% do tempo, então usamos
        # um percentil BAIXO (p5) para capturar os frames de olho fechado.
        open_med = float(np.median(open_ears))
        blink_low = float(np.percentile(blink_ears, 5)) if blink_ears else open_med
        if blink_low < 0.8 * open_med:
            threshold = blink_low + 0.5 * (open_med - blink_low)
            method = "midpoint_open_median_blink_p5"
        else:
            threshold = 0.7 * open_med  # nenhuma piscada clara observada: heurística relativa
            method = "fallback_70pct_open_median"
            print("[pré-checagem] nenhuma piscada clara detectada; usando 70% do EAR aberto.")
        ctx.backend.blink_threshold = threshold
        result.update(open_ear_median=open_med, blink_ear_p5=blink_low,
                      blink_threshold=threshold, blink_threshold_method=method)
    print(f"[pré-checagem] cobertura {result['coverage'] * 100:.1f}% | limiar de piscada {result['blink_threshold']}")
    if result["coverage"] < 0.8:
        print("[pré-checagem] atenção: detecção abaixo de 80%. Melhore iluminação/enquadramento.")
    return result


def collect_targets(ctx: Ctx, phase: str, targets: list[tuple[float, float]], seconds: float,
                    settle: float, transition: float) -> list[dict]:
    """Exibe cada alvo e coleta amostras após o tempo de estabilização."""
    records = []
    for idx, (nx, ny) in enumerate(targets):
        tx, ty = nx * (ctx.width - 1), ny * (ctx.height - 1)
        start = time.perf_counter()
        while (el := time.perf_counter() - start) < seconds:
            canvas = blank_canvas(ctx.width, ctx.height)
            draw_target(ctx.cv2, canvas, tx, ty, el / seconds)
            sample = ctx.pump(canvas, f"{phase} {idx + 1}/{len(targets)}", target=(tx, ty))
            if el < settle:
                continue
            rec = {"phase": phase, "target_idx": idx, "target_x": tx, "target_y": ty,
                   "timestamp_ms": ctx.now_ms(), "valid": sample is not None and not sample.blink,
                   "blink": bool(sample.blink) if sample else False}
            for name, value in zip(FEATURE_NAMES, sample.features if sample else [float("nan")] * 4):
                rec[name] = value
            records.append(rec)
        tstart = time.perf_counter()
        while idx < len(targets) - 1 and time.perf_counter() - tstart < transition:
            ctx.pump(blank_canvas(ctx.width, ctx.height), f"{phase} transicao")
    return records


def fit_calibration(records: list[dict], model: str, ridge: float) -> GazeCalibrator:
    valid = [r for r in records if r["valid"]]
    if not valid:
        raise RuntimeError("Nenhuma amostra válida na calibração.")
    ids = [r["target_idx"] for r in valid]
    feats = np.array([[r[n] for n in FEATURE_NAMES] for r in valid])
    tgts = np.array([[r["target_x"], r["target_y"]] for r in valid])
    feats, tgts, used = clean_calibration_samples(ids, feats, tgts)
    if used < 5:
        raise RuntimeError(f"Calibração falhou: apenas {used} alvos com amostras suficientes (mínimo 5).")
    cal = GazeCalibrator(model, ridge).fit(feats, tgts)
    print(f"[calibração] modelo={model} alvos={used} amostras={len(feats)} RMSE treino={cal.train_rmse_px:.1f}px")
    return cal


def evaluate_validation(records: list[dict], cal: GazeCalibrator, ppd: Optional[float]) -> dict:
    per_target: dict[tuple[float, float], list] = {}
    for r in records:
        key = (r["target_x"], r["target_y"])
        per_target.setdefault(key, [])
        if r["valid"]:
            per_target[key].append(cal.predict([r[n] for n in FEATURE_NAMES]))
    metrics = validation_metrics({k: np.array(v) for k, v in per_target.items()}, ppd)
    metrics["data_loss"] = 1.0 - (sum(r["valid"] for r in records) / len(records)) if records else float("nan")
    msg = f"[validação] acurácia média {metrics['accuracy_mean_px']:.1f}px"
    if "accuracy_mean_deg" in metrics:
        msg += f" ({metrics['accuracy_mean_deg']:.2f}°)"
    print(msg + f" | precisão RMS-S2S {metrics['precision_rms_s2s_px']:.1f}px | perda {metrics['data_loss'] * 100:.1f}%")
    return metrics


def record_stimulus(ctx: Ctx, cal: GazeCalibrator, stimulus, args) -> tuple[list[dict], list[dict], dict]:
    cv2 = ctx.cv2
    if stimulus is not None:
        lb = Letterbox.fit(stimulus.shape[1], stimulus.shape[0], ctx.width, ctx.height)
        base = blank_canvas(ctx.width, ctx.height)
        nw, nh = int(round(stimulus.shape[1] * lb.scale)), int(round(stimulus.shape[0] * lb.scale))
        ox, oy = int(round(lb.off_x)), int(round(lb.off_y))
        base[oy:oy + nh, ox:ox + nw] = cv2.resize(stimulus, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        lb = Letterbox.fit(ctx.width, ctx.height, ctx.width, ctx.height)
        base = blank_canvas(ctx.width, ctx.height)
        put_center(cv2, base, "Explore a tela livremente", ctx.height // 2)

    wait_for_space(ctx, ["Agora sera exibida a tela do teste.", args.task_text or "Observe a tela normalmente."])

    filt = PointFilter(args.filter_min_cutoff, args.filter_beta, enabled=not args.no_filter)
    idt = IDTFixationDetector(args.fixation_dispersion_px, args.fixation_min_duration_ms, args.fixation_max_gap_ms)
    rows, fixations = [], []
    rec_t0 = time.perf_counter()
    stop_reason = "time_limit"

    def add_fix(fix):
        if fix is None:
            return
        sx, sy = lb.screen_to_stimulus(fix.x, fix.y)
        fixations.append({"fixation_id": fix.fixation_id, "start_ms": fix.start_ms, "end_ms": fix.end_ms,
                          "duration_ms": fix.duration_ms, "x": fix.x, "y": fix.y, "stim_x": sx, "stim_y": sy,
                          "samples": fix.samples, "dispersion_px": fix.dispersion_px})

    frame_idx = 0
    try:
        while True:
            elapsed = time.perf_counter() - rec_t0
            if args.max_session_seconds > 0 and elapsed >= args.max_session_seconds:
                break
            frame = ctx.read()
            sample = ctx.backend.process(frame)
            t_ms = elapsed * 1000.0
            valid = sample is not None and not sample.blink
            raw = filt_xy = (float("nan"), float("nan"))
            if valid:
                raw = cal.predict(sample.features)
                filt_xy = filt(raw[0], raw[1], elapsed)
            add_fix(idt.add(t_ms, *filt_xy))
            sx, sy = lb.screen_to_stimulus(*filt_xy)
            feats = sample.features if sample else (float("nan"),) * 4
            rows.append({
                "frame_idx": frame_idx, "timestamp_ms": t_ms, "valid": valid,
                "blink": bool(sample.blink) if sample else False,
                **dict(zip(FEATURE_NAMES, feats)),
                "avg_ear": sample.avg_ear if sample else float("nan"),
                "gaze_raw_x": raw[0], "gaze_raw_y": raw[1], "gaze_x": filt_xy[0], "gaze_y": filt_xy[1],
                "stim_x": sx, "stim_y": sy, "fixation_id": idt.current_fixation_id,
            })
            canvas = base
            if args.show_gaze and valid:
                canvas = base.copy()
                cv2.circle(canvas, (int(filt_xy[0]), int(filt_xy[1])), 14, (0, 255, 0), 2, cv2.LINE_AA)
            ctx.display.show(canvas)
            ctx.display.show_camera(annotate_camera(cv2, frame, sample, "gravacao"))
            if ctx.display.key() in QUIT_KEYS:
                raise SessionAborted()
            frame_idx += 1
    except SessionAborted:
        stop_reason = "user_requested_stop"
    finally:
        add_fix(idt.flush())

    duration_s = time.perf_counter() - rec_t0
    n_valid = sum(r["valid"] for r in rows)
    on_stim = sum(1 for r in rows if r["valid"] and lb.inside_stimulus(r["stim_x"], r["stim_y"]))
    stats = {
        "stop_reason": stop_reason,
        "duration_s": duration_s,
        "frames": len(rows),
        "fps": len(rows) / duration_s if duration_s > 0 else float("nan"),
        "data_loss": 1.0 - n_valid / len(rows) if rows else float("nan"),
        "blink_frames": sum(r["blink"] for r in rows),
        "gaze_on_stimulus_ratio": on_stim / n_valid if n_valid else float("nan"),
        "fixations": len(fixations),
        "letterbox": lb.to_dict(),
    }
    print(f"[gravação] {stats['frames']} frames ({stats['fps']:.1f} fps), perda {stats['data_loss'] * 100:.1f}%, "
          f"{len(fixations)} fixações")
    return rows, fixations, stats


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_session(args, ctx: Ctx, stimulus) -> dict:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ppd = None
    if args.screen_width_cm and args.distance_cm:
        ppd = px_per_degree(ctx.width, args.screen_width_cm, args.distance_cm)

    summary: dict = {
        "participant": args.participant,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": ctx.backend.name,
        "screen": {"width_px": ctx.width, "height_px": ctx.height, "width_cm": args.screen_width_cm,
                   "distance_cm": args.distance_cm, "px_per_degree": ppd},
        "stimulus": str(args.stimulus) if args.stimulus else None,
        "task_text": args.task_text,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "opencv": getattr(ctx.cv2, "__version__", "?")},
        "status": "incomplete",
    }
    target_records: list[dict] = []

    def save(status: str) -> None:
        summary["status"] = status
        write_csv(out / "calibration_samples.csv", target_records)
        (out / "session.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        ctx.display.open()
        if not args.skip_precheck:
            summary["precheck"] = run_precheck(ctx, args.precheck_seconds)

        calib_targets = list(CALIBRATION_POINTS[args.calibration_points])
        rng.shuffle(calib_targets)
        wait_for_space(ctx, ["Calibracao", "Olhe fixamente para o centro de cada circulo.",
                             "Mantenha a cabeca o mais parada possivel."])
        calib = collect_targets(ctx, "calibration", calib_targets, args.target_seconds,
                                args.target_settle_seconds, args.target_transition_seconds)
        target_records.extend(calib)
        cal = fit_calibration(calib, args.calibration_model, args.ridge)
        summary["calibration"] = cal.to_dict()

        if not args.skip_validation:
            val_targets = list(VALIDATION_POINTS)
            rng.shuffle(val_targets)
            wait_for_space(ctx, ["Validacao", "Mais alguns circulos para medir a precisao."])
            val = collect_targets(ctx, "validation", val_targets, args.target_seconds,
                                  args.target_settle_seconds, args.target_transition_seconds)
            target_records.extend(val)
            summary["validation"] = evaluate_validation(val, cal, ppd)

        rows, fixations, stats = record_stimulus(ctx, cal, stimulus, args)
        write_csv(out / "frames.csv", rows)
        write_csv(out / "fixations.csv", fixations)
        summary["recording"] = stats
        save("saved")
    except SessionAborted:
        print("Sessão interrompida pelo usuário; dados parciais salvos.")
        save("aborted")
    except RuntimeError as exc:
        print(f"[erro] {exc}")
        summary["error"] = str(exc)
        save("failed")
    print(f"Saídas em: {out.resolve()}")
    return summary


def detect_screen_size(default: tuple[int, int]) -> tuple[int, int]:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            return size
    except Exception:
        pass
    return default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eye tracking por webcam para estudos de UX")
    g = p.add_argument_group("sessão")
    g.add_argument("--output-dir", type=Path, default=Path("runs/sessao"))
    g.add_argument("--participant", default="P00", help="identificador anônimo do participante")
    g.add_argument("--stimulus", type=Path, help="imagem exibida na gravação (ex.: print de um site)")
    g.add_argument("--task-text", default="", help="instrução da tarefa exibida antes do estímulo")
    g.add_argument("--max-session-seconds", type=float, default=30.0, help="0 = até apertar Q")
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("hardware")
    g.add_argument("--camera-index", type=int, default=0)
    g.add_argument("--backend", choices=["auto", "mediapipe", "opencv"], default="auto")
    g.add_argument("--screen-size", default="", help="LARGURAxALTURA em px (padrão: detecta)")
    g.add_argument("--screen-width-cm", type=float, default=0.0, help="largura física da tela (para graus)")
    g.add_argument("--distance-cm", type=float, default=0.0, help="distância olho-tela (para graus)")
    g.add_argument("--show-camera", action="store_true", help="mostra janela da câmera (pesquisador)")
    g = p.add_argument_group("calibração")
    g.add_argument("--skip-precheck", action="store_true")
    g.add_argument("--precheck-seconds", type=float, default=6.0)
    g.add_argument("--calibration-points", choices=list(CALIBRATION_POINTS), default="9")
    g.add_argument("--calibration-model", choices=["linear", "poly2"], default="poly2")
    g.add_argument("--ridge", type=float, default=1e-2)
    g.add_argument("--target-seconds", type=float, default=2.0)
    g.add_argument("--target-settle-seconds", type=float, default=0.7)
    g.add_argument("--target-transition-seconds", type=float, default=0.3)
    g.add_argument("--skip-validation", action="store_true")
    g = p.add_argument_group("processamento")
    g.add_argument("--no-filter", action="store_true", help="desliga o filtro One Euro")
    g.add_argument("--filter-min-cutoff", type=float, default=0.8)
    g.add_argument("--filter-beta", type=float, default=0.005)
    g.add_argument("--fixation-dispersion-px", type=float, default=120.0)
    g.add_argument("--fixation-min-duration-ms", type=float, default=100.0)
    g.add_argument("--fixation-max-gap-ms", type=float, default=100.0)
    g.add_argument("--show-gaze", action="store_true",
                   help="mostra o cursor do olhar sobre o estímulo (só para demonstração; enviesa testes)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    import cv2

    if args.screen_size:
        width, height = (int(v) for v in args.screen_size.lower().split("x"))
    else:
        width, height = detect_screen_size((1920, 1080))

    stimulus = None
    if args.stimulus:
        stimulus = cv2.imread(str(args.stimulus))
        if stimulus is None:
            raise SystemExit(f"Não foi possível ler o estímulo: {args.stimulus}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a webcam.")
    backend = choose_backend(args.backend, cv2)
    display = CvDisplay(cv2, width, height, args.show_camera)
    print(f"Backend: {backend.name} | tela {width}x{height}")
    try:
        run_session(args, Ctx(cv2, cap, backend, display, width, height, time.perf_counter()), stimulus)
    finally:
        cap.release()
        backend.close()
        display.close()


if __name__ == "__main__":
    main()
