from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


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


@dataclass
class FixationState:
    fixation_id: int = 0
    start_ms: Optional[float] = None
    points: list[tuple[float, float]] | None = None
    active_id: int = -1

    def __post_init__(self) -> None:
        if self.points is None:
            self.points = []


def to_pixel(point, width: int, height: int) -> tuple[float, float]:
    return float(point.x * width), float(point.y * height)


def ratio_between(v: float, a: float, b: float) -> float:
    lo, hi = min(a, b), max(a, b)
    if math.isclose(hi, lo):
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def eye_aspect_ratio(top: tuple[float, float], bottom: tuple[float, float],
                     inner: tuple[float, float], outer: tuple[float, float]) -> float:
    vertical = math.dist(top, bottom)
    horizontal = math.dist(inner, outer)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.array(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def update_fixation(
    state: FixationState,
    point_px: tuple[float, float],
    timestamp_ms: float,
    threshold_px: float,
    min_duration_ms: float,
) -> tuple[int, Optional[dict]]:
    """Atualiza estado de fixação.

    Retorna:
        fixation_id_ativo (ou -1 se não consolidado), fixation_finalizada (ou None)
    """
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eye tracking por webcam para UX")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default_session"))
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--fixation-threshold-px", type=float, default=60.0)
    parser.add_argument("--fixation-min-duration-ms", type=float, default=180.0)
    parser.add_argument("--blink-ear-threshold", type=float, default=0.21)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    mp_face_mesh = mp.solutions.face_mesh
    frame_rows: list[dict] = []
    fixation_rows: list[dict] = []
    fix_state = FixationState()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            timestamp_ms = (time.perf_counter() - t0) * 1000.0

            gaze_x = np.nan
            gaze_y = np.nan
            blink = False
            left_ear = np.nan
            right_ear = np.nan
            avg_ear = np.nan
            fixation_id = -1

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                left_iris_pts = [to_pixel(landmarks[i], w, h) for i in LEFT_IRIS]
                right_iris_pts = [to_pixel(landmarks[i], w, h) for i in RIGHT_IRIS]
                left_iris_center = centroid(left_iris_pts)
                right_iris_center = centroid(right_iris_pts)

                left_inner = to_pixel(landmarks[LEFT_EYE_INNER], w, h)
                left_outer = to_pixel(landmarks[LEFT_EYE_OUTER], w, h)
                left_top = to_pixel(landmarks[LEFT_EYE_TOP], w, h)
                left_bottom = to_pixel(landmarks[LEFT_EYE_BOTTOM], w, h)

                right_inner = to_pixel(landmarks[RIGHT_EYE_INNER], w, h)
                right_outer = to_pixel(landmarks[RIGHT_EYE_OUTER], w, h)
                right_top = to_pixel(landmarks[RIGHT_EYE_TOP], w, h)
                right_bottom = to_pixel(landmarks[RIGHT_EYE_BOTTOM], w, h)

                left_ratio_x = ratio_between(left_iris_center[0], left_outer[0], left_inner[0])
                right_ratio_x = ratio_between(right_iris_center[0], right_outer[0], right_inner[0])
                left_ratio_y = ratio_between(left_iris_center[1], left_top[1], left_bottom[1])
                right_ratio_y = ratio_between(right_iris_center[1], right_top[1], right_bottom[1])

                gaze_x = float(np.clip((left_ratio_x + right_ratio_x) / 2.0, 0.0, 1.0))
                gaze_y = float(np.clip((left_ratio_y + right_ratio_y) / 2.0, 0.0, 1.0))

                left_ear = eye_aspect_ratio(left_top, left_bottom, left_inner, left_outer)
                right_ear = eye_aspect_ratio(right_top, right_bottom, right_inner, right_outer)
                avg_ear = (left_ear + right_ear) / 2.0
                blink = avg_ear < args.blink_ear_threshold

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
                }
            )

            if args.show_window:
                cv2.putText(
                    frame,
                    f"Blink: {blink}  EAR: {avg_ear:.3f}" if not np.isnan(avg_ear) else "Blink: N/A",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Fixation ID: {fixation_id}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("Eye Tracking UX", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
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
    pd.DataFrame(frame_rows).to_csv(frames_path, index=False)
    pd.DataFrame(fixation_rows).to_csv(fixations_path, index=False)

    print(f"Frames salvos em: {frames_path}")
    print(f"Fixações salvas em: {fixations_path}")


if __name__ == "__main__":
    main()
