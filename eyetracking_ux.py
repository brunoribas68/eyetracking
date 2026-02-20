from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def apply_gaze_gain(value: float, gain: float) -> float:
    if math.isnan(value):
        return value
    return clip(0.5 + (value - 0.5) * gain)


@dataclass
class FixationState:
    fixation_id: int = 0
    start_ms: Optional[float] = None
    points: list[tuple[float, float]] | None = None
    active_id: int = -1

    def __post_init__(self) -> None:
        if self.points is None:
            self.points = []


@dataclass
class GazeSample:
    gaze_x: float
    gaze_y: float
    overlay_x: float
    overlay_y: float
    blink: bool
    left_ear: float
    right_ear: float
    avg_ear: float


@dataclass
class SessionConfig:
    training_display_target: str
    gaze_overlay_mode: str


@dataclass
class ScreenCalibration:
    left_x: float
    right_x: float
    top_y: float
    bottom_y: float
    corner_samples: dict[str, dict[str, float]]


def ratio_between(v: float, a: float, b: float) -> float:
    lo, hi = min(a, b), max(a, b)
    if math.isclose(hi, lo):
        return 0.5
    return clip((v - lo) / (hi - lo), 0.0, 1.0)


def centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    n = max(len(points), 1)
    return x_sum / n, y_sum / n


def update_fixation(
    state: FixationState,
    point_px: tuple[float, float],
    timestamp_ms: float,
    threshold_px: float,
    min_duration_ms: float,
) -> tuple[int, Optional[dict]]:
    finalized = None

    if not state.points:
        state.start_ms = timestamp_ms
        state.points = [point_px]
        state.active_id = -1
        return state.active_id, finalized

    cx, cy = centroid(state.points)
    dist = math.dist((cx, cy), point_px)

    if dist <= threshold_px:
        state.points.append(point_px)
        duration = timestamp_ms - (state.start_ms or timestamp_ms)
        if duration >= min_duration_ms and state.active_id == -1:
            state.active_id = state.fixation_id
        return state.active_id, finalized

    duration = timestamp_ms - (state.start_ms or timestamp_ms)
    if duration >= min_duration_ms:
        fx, fy = centroid(state.points)
        finalized = {
            "fixation_id": state.fixation_id,
            "start_ms": state.start_ms,
            "end_ms": timestamp_ms,
            "duration_ms": duration,
            "centroid_x": fx,
            "centroid_y": fy,
            "samples": len(state.points),
        }
        state.fixation_id += 1

    state.start_ms = timestamp_ms
    state.points = [point_px]
    state.active_id = -1
    return state.active_id, finalized


class MediaPipeBackend:
    def __init__(self) -> None:
        import mediapipe as mp

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def to_pixel(point, width: int, height: int) -> tuple[float, float]:
        return float(point.x * width), float(point.y * height)

    @staticmethod
    def eye_aspect_ratio(top: tuple[float, float], bottom: tuple[float, float],
                         inner: tuple[float, float], outer: tuple[float, float]) -> float:
        vertical = math.dist(top, bottom)
        horizontal = math.dist(inner, outer)
        if horizontal <= 1e-6:
            return 0.0
        return vertical / horizontal

    def process(self, frame_bgr, cv2_module, blink_ear_threshold: float) -> Optional[GazeSample]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        left_iris_pts = [self.to_pixel(landmarks[i], w, h) for i in LEFT_IRIS]
        right_iris_pts = [self.to_pixel(landmarks[i], w, h) for i in RIGHT_IRIS]
        left_iris_center = centroid(left_iris_pts)
        right_iris_center = centroid(right_iris_pts)

        left_inner = self.to_pixel(landmarks[LEFT_EYE_INNER], w, h)
        left_outer = self.to_pixel(landmarks[LEFT_EYE_OUTER], w, h)
        left_top = self.to_pixel(landmarks[LEFT_EYE_TOP], w, h)
        left_bottom = self.to_pixel(landmarks[LEFT_EYE_BOTTOM], w, h)

        right_inner = self.to_pixel(landmarks[RIGHT_EYE_INNER], w, h)
        right_outer = self.to_pixel(landmarks[RIGHT_EYE_OUTER], w, h)
        right_top = self.to_pixel(landmarks[RIGHT_EYE_TOP], w, h)
        right_bottom = self.to_pixel(landmarks[RIGHT_EYE_BOTTOM], w, h)

        left_ratio_x = ratio_between(left_iris_center[0], left_outer[0], left_inner[0])
        right_ratio_x = ratio_between(right_iris_center[0], right_outer[0], right_inner[0])
        left_ratio_y = ratio_between(left_iris_center[1], left_top[1], left_bottom[1])
        right_ratio_y = ratio_between(right_iris_center[1], right_top[1], right_bottom[1])

        gaze_x = clip((left_ratio_x + right_ratio_x) / 2.0)
        gaze_y = clip((left_ratio_y + right_ratio_y) / 2.0)

        left_ear = self.eye_aspect_ratio(left_top, left_bottom, left_inner, left_outer)
        right_ear = self.eye_aspect_ratio(right_top, right_bottom, right_inner, right_outer)
        avg_ear = (left_ear + right_ear) / 2.0

        overlay_x = right_iris_center[0]
        overlay_y = right_iris_center[1]
        return GazeSample(gaze_x, gaze_y, overlay_x, overlay_y, avg_ear < blink_ear_threshold, left_ear, right_ear, avg_ear)

    def close(self) -> None:
        self.face_mesh.close()


class OpenCVBackend:
    def __init__(self, cv2_module) -> None:
        self.cv2 = cv2_module
        self.face = cv2_module.CascadeClassifier(
            cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eyes = cv2_module.CascadeClassifier(
            cv2_module.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    @staticmethod
    def _is_reasonable_eye(ex: int, ey: int, ew: int, eh: int, roi_w: int, roi_h: int) -> bool:
        if ew <= 0 or eh <= 0:
            return False
        aspect = eh / max(ew, 1)
        center_x = ex + ew / 2.0
        center_y = ey + eh / 2.0
        if not (0.12 <= aspect <= 0.9):
            return False
        if center_y > roi_h * 0.72:
            return False
        if center_x < roi_w * 0.08 or center_x > roi_w * 0.92:
            return False
        return True

    def _pupil_ratio(self, gray_eye) -> tuple[float, float]:
        blur = self.cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, thresh = self.cv2.threshold(blur, 0, 255, self.cv2.THRESH_BINARY_INV + self.cv2.THRESH_OTSU)
        m = self.cv2.moments(thresh)
        h, w = gray_eye.shape[:2]
        if m["m00"] < 1:
            return 0.5, 0.5
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        return clip(cx / max(w, 1)), clip(cy / max(h, 1))

    def process(self, frame_bgr, cv2_module, blink_ear_threshold: float) -> Optional[GazeSample]:
        gray = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y:y + int(h * 0.6), x:x + w]
        eyes = self.eyes.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        if len(eyes) < 1:
            return None

        filtered_eyes = [e for e in eyes if self._is_reasonable_eye(e[0], e[1], e[2], e[3], w, roi.shape[0])]
        eye_candidates = filtered_eyes if len(filtered_eyes) > 0 else list(eyes)

        selected = sorted(eye_candidates, key=lambda r: r[2] * r[3], reverse=True)
        if len(selected) >= 2:
            leftmost = min(selected, key=lambda r: r[0])
            rightmost = max(selected, key=lambda r: r[0])
            horizontal_gap = abs((rightmost[0] + rightmost[2] / 2.0) - (leftmost[0] + leftmost[2] / 2.0))
            min_gap = 0.18 * w
            if horizontal_gap >= min_gap:
                selected = [leftmost, rightmost]
            else:
                selected = [selected[0]]
        else:
            selected = selected[:1]

        x_ratios: list[float] = []
        y_ratios: list[float] = []
        ears: list[float] = []
        overlay_points: list[tuple[float, float]] = []

        for ex, ey, ew, eh in selected:
            eye_roi = roi[ey:ey + eh, ex:ex + ew]
            if eye_roi.size == 0:
                continue
            rx, ry = self._pupil_ratio(eye_roi)
            x_ratios.append(rx)
            y_ratios.append(ry)
            ears.append(eh / max(ew, 1))
            overlay_points.append((x + ex + rx * ew, y + ey + ry * eh))

        if not x_ratios:
            return None

        gaze_x = clip(safe_mean(x_ratios))
        gaze_y = clip(safe_mean(y_ratios))
        overlay_x, overlay_y = min(overlay_points, key=lambda p: p[1])
        avg_ear = safe_mean(ears)
        left_ear = ears[0] if len(ears) > 0 else float("nan")
        right_ear = ears[1] if len(ears) > 1 else float("nan")
        blink = (not math.isnan(avg_ear)) and (avg_ear < blink_ear_threshold)
        return GazeSample(gaze_x, gaze_y, overlay_x, overlay_y, blink, left_ear, right_ear, avg_ear)

    def close(self) -> None:
        return None


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    p = clip(p, 0.0, 1.0)
    idx = int(round((len(ordered) - 1) * p))
    return ordered[idx]


def run_precalibration_check(
    cap,
    backend,
    cv2_module,
    seconds: float,
    show_window: bool,
    blink_ear_threshold: float,
) -> float:
    print("\n[Pré-calibragem] Iniciando validação de câmera/piscada...")
    print("- Primeiro mantenha os olhos abertos.")
    print("- Depois pisque naturalmente por alguns segundos.\n")

    start = time.perf_counter()
    open_phase = max(seconds * 0.5, 0.1)
    open_ears: list[float] = []
    blink_ears: list[float] = []
    frames_seen = 0
    frames_with_face = 0

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break

        ok, frame = cap.read()
        if not ok:
            continue
        frames_seen += 1

        sample = backend.process(frame, cv2_module, blink_ear_threshold)
        phase_open = elapsed < open_phase
        phase_text = "Olhos abertos" if phase_open else "Piscar natural"

        if sample is not None and not math.isnan(sample.avg_ear):
            frames_with_face += 1
            if phase_open:
                open_ears.append(sample.avg_ear)
            else:
                blink_ears.append(sample.avg_ear)

        if show_window:
            cv2_module.putText(frame, "Pre-calibragem", (20, 30), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2_module.putText(frame, phase_text, (20, 60), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if sample is not None and not math.isnan(sample.avg_ear):
                cv2_module.putText(frame, f"EAR: {sample.avg_ear:.3f}", (20, 90), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2_module.putText(frame, "Rosto/olhos nao detectados", (20, 90), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2_module.imshow("Eye Tracking UX", frame)
            if cv2_module.waitKey(1) & 0xFF == ord("q"):
                break

    coverage = 0.0 if frames_seen == 0 else frames_with_face / frames_seen
    open_median = percentile(open_ears, 0.5)
    blink_low = percentile(blink_ears, 0.2)

    calibrated_threshold = blink_ear_threshold
    if not math.isnan(open_median) and not math.isnan(blink_low) and open_median > blink_low:
        calibrated_threshold = (open_median + blink_low) / 2.0

    print(f"[Pré-calibragem] Cobertura de detecção: {coverage * 100:.1f}% ({frames_with_face}/{frames_seen}).")
    if coverage < 0.6:
        print("[Pré-calibragem] Atenção: detecção baixa. Melhore iluminação e enquadramento antes de calibrar.")
    if math.isnan(open_median) or math.isnan(blink_low):
        print("[Pré-calibragem] Amostras insuficientes para ajustar EAR. Mantendo threshold atual.")
    else:
        print(
            f"[Pré-calibragem] EAR aberto mediano={open_median:.3f}, EAR baixo de piscada={blink_low:.3f}, "
            f"novo threshold={calibrated_threshold:.3f}."
        )

    return calibrated_threshold


def map_gaze_with_calibration(gaze_x: float, gaze_y: float, calibration: Optional[ScreenCalibration]) -> tuple[float, float]:
    if calibration is None:
        return gaze_x, gaze_y
    x_den = calibration.right_x - calibration.left_x
    y_den = calibration.bottom_y - calibration.top_y
    if math.isclose(x_den, 0.0) or math.isclose(y_den, 0.0):
        return gaze_x, gaze_y
    mapped_x = clip((gaze_x - calibration.left_x) / x_den)
    mapped_y = clip((gaze_y - calibration.top_y) / y_den)
    return mapped_x, mapped_y


def run_corner_training(
    cap,
    backend,
    cv2_module,
    blink_ear_threshold: float,
    seconds_per_corner: float,
) -> Optional[ScreenCalibration]:
    corners = [
        ("top_right", 0.90, 0.10, "Olhe para o canto SUPERIOR DIREITO"),
        ("bottom_right", 0.90, 0.90, "Olhe para o canto INFERIOR DIREITO"),
        ("top_left", 0.10, 0.10, "Olhe para o canto SUPERIOR ESQUERDO"),
        ("bottom_left", 0.10, 0.90, "Olhe para o canto INFERIOR ESQUERDO"),
    ]
    print("\n[Treinamento] Iniciando calibração por cantos da tela...")
    print("[Treinamento] Sequência: superior direito -> inferior direito -> superior esquerdo -> inferior esquerdo.")

    samples: dict[str, dict[str, list[float]]] = {
        name: {"x": [], "y": []} for name, _, _, _ in corners
    }

    for name, tx, ty, instruction in corners:
        phase_start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - phase_start
            if elapsed >= seconds_per_corner:
                break

            ok, frame = cap.read()
            if not ok:
                continue

            sample = backend.process(frame, cv2_module, blink_ear_threshold)
            if sample is not None and not sample.blink:
                samples[name]["x"].append(sample.gaze_x)
                samples[name]["y"].append(sample.gaze_y)

            h, w = frame.shape[:2]
            cv2_module.putText(frame, "Treinamento de tela", (20, 30), cv2_module.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2_module.putText(frame, instruction, (20, 60), cv2_module.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2_module.putText(frame, f"Tempo restante: {max(0.0, seconds_per_corner - elapsed):.1f}s", (20, 90), cv2_module.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2_module.circle(frame, (int(tx * (w - 1)), int(ty * (h - 1))), 14, (0, 255, 0), -1)
            cv2_module.imshow("Eye Tracking UX", frame)

            gaze_screen = frame.copy()
            gaze_screen[:] = 20
            cv2_module.putText(gaze_screen, "Gaze Screen (training)", (20, 30), cv2_module.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            sh, sw = gaze_screen.shape[:2]
            cv2_module.circle(gaze_screen, (int(tx * (sw - 1)), int(ty * (sh - 1))), 14, (0, 255, 0), -1)
            cv2_module.imshow("Eye Tracking UX - Gaze Screen", gaze_screen)

            key = cv2_module.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                print("[Treinamento] Treinamento interrompido pelo usuário.")
                return None

    corner_stats: dict[str, dict[str, float]] = {}
    for name, _, _, _ in corners:
        x_med = percentile(samples[name]["x"], 0.5)
        y_med = percentile(samples[name]["y"], 0.5)
        if math.isnan(x_med) or math.isnan(y_med):
            print(f"[Treinamento] Falha: amostras insuficientes no canto '{name}'.")
            return None
        corner_stats[name] = {"gaze_x": x_med, "gaze_y": y_med}

    left_x = safe_mean([corner_stats["top_left"]["gaze_x"], corner_stats["bottom_left"]["gaze_x"]])
    right_x = safe_mean([corner_stats["top_right"]["gaze_x"], corner_stats["bottom_right"]["gaze_x"]])
    top_y = safe_mean([corner_stats["top_left"]["gaze_y"], corner_stats["top_right"]["gaze_y"]])
    bottom_y = safe_mean([corner_stats["bottom_left"]["gaze_y"], corner_stats["bottom_right"]["gaze_y"]])

    if math.isclose(left_x, right_x) or math.isclose(top_y, bottom_y):
        print("[Treinamento] Falha: calibração degenerada (faixa de olhos muito pequena).")
        return None

    print("[Treinamento] Calibração concluída com sucesso.")
    return ScreenCalibration(
        left_x=left_x,
        right_x=right_x,
        top_y=top_y,
        bottom_y=bottom_y,
        corner_samples=corner_stats,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eye tracking por webcam para UX")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default_session"))
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--backend", choices=["auto", "mediapipe", "opencv"], default="auto")
    parser.add_argument("--fixation-threshold-px", type=float, default=60.0)
    parser.add_argument("--fixation-min-duration-ms", type=float, default=180.0)
    parser.add_argument("--blink-ear-threshold", type=float, default=0.21)
    parser.add_argument("--precheck-seconds", type=float, default=8.0)
    parser.add_argument("--skip-precheck", action="store_true")
    parser.add_argument("--max-session-seconds", type=float, default=0.0)
    parser.add_argument(
        "--training-display-target",
        choices=["main", "secondary", "remote"],
        default="main",
        help="Prepara o modo de treino para exibir olhar na tela principal, secundária ou fluxo remoto.",
    )
    parser.add_argument(
        "--gaze-overlay-mode",
        choices=["cursor", "heatmap_stub"],
        default="cursor",
        help="Estratégia base para visualização de treino/UX (cursor atual ou base para heatmap).",
    )
    parser.add_argument(
        "--gaze-gain-x",
        type=float,
        default=1.0,
        help="Ganho horizontal do ponto de olhar projetado na tela (1.0 = sem ganho).",
    )
    parser.add_argument(
        "--gaze-gain-y",
        type=float,
        default=1.0,
        help="Ganho vertical do ponto de olhar projetado na tela (1.0 = sem ganho).",
    )
    parser.add_argument("--skip-corner-training", action="store_true", help="Pula o treinamento de cantos da tela antes da coleta.")
    parser.add_argument("--corner-seconds", type=float, default=2.0, help="Segundos de coleta por canto durante o treinamento.")
    return parser


def choose_backend(name: str, cv2_module):
    if name in {"auto", "mediapipe"}:
        try:
            return MediaPipeBackend(), "mediapipe"
        except Exception:
            if name == "mediapipe":
                raise
    return OpenCVBackend(cv2_module), "opencv"


def main() -> None:
    args = build_parser().parse_args()
    import cv2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    backend, backend_name = choose_backend(args.backend, cv2)
    print(f"Backend selecionado: {backend_name}")

    blink_ear_threshold = args.blink_ear_threshold
    if not args.skip_precheck and args.precheck_seconds > 0:
        blink_ear_threshold = run_precalibration_check(
            cap,
            backend,
            cv2,
            args.precheck_seconds,
            args.show_window,
            blink_ear_threshold,
        )
    print(f"Threshold de piscada em uso: {blink_ear_threshold:.3f}")

    screen_calibration: Optional[ScreenCalibration] = None
    if not args.skip_corner_training:
        if not args.show_window:
            print("[Treinamento] --show-window não habilitado. Pulando treinamento de cantos.")
        else:
            screen_calibration = run_corner_training(
                cap,
                backend,
                cv2,
                blink_ear_threshold,
                max(args.corner_seconds, 0.5),
            )
            if screen_calibration is None:
                print("[Treinamento] Calibração não concluída; usando mapeamento padrão.")

    frame_rows: list[dict] = []
    fixation_rows: list[dict] = []
    fix_state = FixationState()
    session_config = SessionConfig(
        training_display_target=args.training_display_target,
        gaze_overlay_mode=args.gaze_overlay_mode,
    )

    frame_idx = 0
    t0 = time.perf_counter()
    stop_reason = "camera_ended"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                stop_reason = "camera_ended"
                break

            if args.max_session_seconds > 0 and (time.perf_counter() - t0) >= args.max_session_seconds:
                stop_reason = "max_session_seconds"
                break

            h, w = frame.shape[:2]
            timestamp_ms = (time.perf_counter() - t0) * 1000.0
            sample = backend.process(frame, cv2, blink_ear_threshold)

            gaze_x = float("nan")
            gaze_y = float("nan")
            blink = False
            left_ear = float("nan")
            right_ear = float("nan")
            avg_ear = float("nan")
            fixation_id = -1

            if sample is not None:
                mapped_x, mapped_y = map_gaze_with_calibration(sample.gaze_x, sample.gaze_y, screen_calibration)
                gaze_x = apply_gaze_gain(mapped_x, max(args.gaze_gain_x, 0.1))
                gaze_y = apply_gaze_gain(mapped_y, max(args.gaze_gain_y, 0.1))
                blink = sample.blink
                left_ear = sample.left_ear
                right_ear = sample.right_ear
                avg_ear = sample.avg_ear

                point_px = (gaze_x * w, gaze_y * h)
                fixation_id, finalized = update_fixation(
                    fix_state,
                    point_px,
                    timestamp_ms,
                    args.fixation_threshold_px,
                    args.fixation_min_duration_ms,
                )
                if finalized is not None:
                    fixation_rows.append(finalized)

                if args.show_window and args.gaze_overlay_mode == "cursor":
                    cv2.circle(frame, (int(sample.overlay_x), int(sample.overlay_y)), 8, (0, 255, 0), -1)
                    cv2.circle(frame, (int(point_px[0]), int(point_px[1])), 6, (0, 255, 255), 2)
            elif args.show_window:
                cv2.putText(frame, "Olhos nao detectados (ajuste luz/enquadramento)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "blink": blink,
                    "left_ear": left_ear,
                    "right_ear": right_ear,
                    "avg_ear": avg_ear,
                    "fixation_id": fixation_id,
                    "backend": backend_name,
                }
            )

            if args.show_window:
                cv2.putText(frame, f"Backend: {backend_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Blink: {blink}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Target: {session_config.training_display_target}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Eye Tracking UX", frame)

                if session_config.training_display_target in {"secondary", "remote"}:
                    gaze_screen = frame.copy()
                    gaze_screen[:] = 20
                    screen_h, screen_w = gaze_screen.shape[:2]
                    cv2.putText(
                        gaze_screen,
                        f"Gaze Screen ({session_config.training_display_target})",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.rectangle(gaze_screen, (0, 0), (screen_w - 1, screen_h - 1), (80, 80, 80), 2)
                    if sample is not None:
                        sx = int(clip(gaze_x) * (screen_w - 1))
                        sy = int(clip(gaze_y) * (screen_h - 1))
                        cv2.circle(gaze_screen, (sx, sy), 12, (0, 255, 0), -1)
                    else:
                        cv2.putText(gaze_screen, "Sem deteccao de olhar", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                    cv2.imshow("Eye Tracking UX - Gaze Screen", gaze_screen)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    stop_reason = "user_requested_stop"
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\nInterrupção detectada. Salvando sessão parcial...")

    cap.release()
    backend.close()
    if args.show_window:
        cv2.destroyAllWindows()

    if fix_state.points and fix_state.start_ms is not None and frame_rows:
        end_ms = frame_rows[-1]["timestamp_ms"]
        duration = end_ms - fix_state.start_ms
        if duration >= args.fixation_min_duration_ms:
            fx, fy = centroid(fix_state.points)
            fixation_rows.append(
                {
                    "fixation_id": fix_state.fixation_id,
                    "start_ms": fix_state.start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration,
                    "centroid_x": fx,
                    "centroid_y": fy,
                    "samples": len(fix_state.points),
                }
            )

    frames_path = args.output_dir / "frames.csv"
    fixations_path = args.output_dir / "fixations.csv"
    session_path = args.output_dir / "session.json"

    write_csv(
        frames_path,
        frame_rows,
        [
            "frame_idx",
            "timestamp_ms",
            "gaze_x",
            "gaze_y",
            "blink",
            "left_ear",
            "right_ear",
            "avg_ear",
            "fixation_id",
            "backend",
        ],
    )
    write_csv(
        fixations_path,
        fixation_rows,
        ["fixation_id", "start_ms", "end_ms", "duration_ms", "centroid_x", "centroid_y", "samples"],
    )

    session_summary = {
        "status": "saved",
        "stop_reason": stop_reason,
        "backend": backend_name,
        "blink_ear_threshold": blink_ear_threshold,
        "total_frames": len(frame_rows),
        "total_fixations": len(fixation_rows),
        "training_prep": {
            "display_target": session_config.training_display_target,
            "overlay_mode": session_config.gaze_overlay_mode,
            "next_step": "Integrar render em tela secundária/fluxo remoto para testes de UX.",
        },
        "screen_calibration": None
        if screen_calibration is None
        else {
            "left_x": screen_calibration.left_x,
            "right_x": screen_calibration.right_x,
            "top_y": screen_calibration.top_y,
            "bottom_y": screen_calibration.bottom_y,
            "corners": screen_calibration.corner_samples,
        },
    }
    session_path.write_text(json.dumps(session_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Frames salvos em: {frames_path}")
    print(f"Fixações salvas em: {fixations_path}")
    print(f"Resumo da sessão salvo em: {session_path}")


if __name__ == "__main__":
    main()
