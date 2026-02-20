from __future__ import annotations

import argparse
import csv
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
    blink: bool
    left_ear: float
    right_ear: float
    avg_ear: float


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

        return GazeSample(gaze_x, gaze_y, avg_ear < blink_ear_threshold, left_ear, right_ear, avg_ear)

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
        eyes = self.eyes.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        if len(eyes) < 1:
            return None

        selected = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]
        x_ratios: list[float] = []
        y_ratios: list[float] = []
        ears: list[float] = []

        for ex, ey, ew, eh in selected:
            eye_roi = roi[ey:ey + eh, ex:ex + ew]
            if eye_roi.size == 0:
                continue
            rx, ry = self._pupil_ratio(eye_roi)
            x_ratios.append(rx)
            y_ratios.append(ry)
            ears.append(eh / max(ew, 1))

        if not x_ratios:
            return None

        gaze_x = clip(safe_mean(x_ratios))
        gaze_y = clip(safe_mean(y_ratios))
        avg_ear = safe_mean(ears)
        left_ear = ears[0] if len(ears) > 0 else float("nan")
        right_ear = ears[1] if len(ears) > 1 else float("nan")
        blink = (not math.isnan(avg_ear)) and (avg_ear < blink_ear_threshold)
        return GazeSample(gaze_x, gaze_y, blink, left_ear, right_ear, avg_ear)

    def close(self) -> None:
        return None


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eye tracking por webcam para UX")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default_session"))
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--backend", choices=["auto", "mediapipe", "opencv"], default="auto")
    parser.add_argument("--fixation-threshold-px", type=float, default=60.0)
    parser.add_argument("--fixation-min-duration-ms", type=float, default=180.0)
    parser.add_argument("--blink-ear-threshold", type=float, default=0.21)
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

    frame_rows: list[dict] = []
    fixation_rows: list[dict] = []
    fix_state = FixationState()

    frame_idx = 0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            timestamp_ms = (time.perf_counter() - t0) * 1000.0
            sample = backend.process(frame, cv2, args.blink_ear_threshold)

            gaze_x = float("nan")
            gaze_y = float("nan")
            blink = False
            left_ear = float("nan")
            right_ear = float("nan")
            avg_ear = float("nan")
            fixation_id = -1

            if sample is not None:
                gaze_x = sample.gaze_x
                gaze_y = sample.gaze_y
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

                if args.show_window:
                    cv2.circle(frame, (int(point_px[0]), int(point_px[1])), 8, (0, 255, 0), -1)

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
                cv2.imshow("Eye Tracking UX", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário (Ctrl+C). Salvando sessão...")

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

    print(f"Frames salvos em: {frames_path}")
    print(f"Fixações salvas em: {fixations_path}")


if __name__ == "__main__":
    main()
