"""Análise de sessões gravadas com eyetracking_ux.py.

Gera, para uma ou mais sessões:
- heatmap_<participante>.png e scanpath_<participante>.png
- group_heatmap.png (todas as sessões somadas, como nos mapas de Bojko)
- summary.csv      (qualidade dos dados + estatísticas de fixação por sessão)
- aoi_metrics.csv  (TTFF, nº de fixações, tempo de permanência, visitas por AOI)
- model_comparison.csv (--compare-models: reajusta linear vs. poly2 offline
                        usando os pontos de calibração e avalia nos de validação)

- --reprocess: recalcula piscadas, calibração, validação e fixações a partir
               dos dados brutos salvos, com o código e os parâmetros atuais
               (útil para sessões gravadas antes de uma correção).

Uso:
    python analyze_session.py runs/P01 runs/P02 --aois aois.json --out resultados --compare-models
    python analyze_session.py runs/P01 --reprocess --out resultados

Formato do aois.json (coordenadas em pixels da imagem do estímulo):
    [{"name": "menu", "x": 20, "y": 80, "w": 160, "h": 220}, ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from gaze_core import (FEATURE_NAMES, CALIBRATION_MODELS, EarModel, GazeCalibrator, Letterbox,
                       ProcessingParams, clean_calibration_samples, label_target_blinks,
                       process_recording, validation_metrics)


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class Session:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
        self.frames = read_csv(path / "frames.csv")
        self.fixations = [f for f in read_csv(path / "fixations.csv")
                          if math.isfinite(f.get("stim_x", math.nan)) and math.isfinite(f.get("stim_y", math.nan))]
        self.calib = read_csv(path / "calibration_samples.csv")
        self.id = str(self.meta.get("participant") or path.name)


def load_stimulus(sessions: list[Session], override: Optional[Path]) -> np.ndarray:
    path = override or (Path(sessions[0].meta["stimulus"]) if sessions[0].meta.get("stimulus") else None)
    if path is not None:
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"Não foi possível ler o estímulo {path}")
        return img
    lb = sessions[0].meta.get("recording", {}).get("letterbox", {})
    w, h = int(lb.get("stim_w", 1920)), int(lb.get("stim_h", 1080))
    return np.full((h, w, 3), 230, np.uint8)


# ---------------------------------------------------------------------------
# visualizações
# ---------------------------------------------------------------------------
def density_map(fixations: list[dict], shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Soma gaussianas nas fixações, ponderadas pela duração."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    for f in fixations:
        x, y = int(round(f["stim_x"])), int(round(f["stim_y"]))
        if 0 <= x < w and 0 <= y < h:
            acc[y, x] += float(f["duration_ms"])
    if acc.max() <= 0:
        return acc
    k = int(6 * sigma) | 1
    return cv2.GaussianBlur(acc, (k, k), sigma)


def render_heatmap(stimulus: np.ndarray, density: np.ndarray, min_level: float = 0.05) -> np.ndarray:
    # página em cinza esmaecido (como em Bojko) para o calor ficar legível em qualquer site
    gray = cv2.cvtColor(cv2.cvtColor(stimulus, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = (gray.astype(np.float32) * 0.45 + 70).astype(np.uint8)
    if density.max() <= 0:
        return out
    norm = density / density.max()
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip((norm - min_level) / (1 - min_level), 0, 1)[..., None] ** 0.6 * 0.75
    out = (out * (1 - alpha) + colored * alpha).astype(np.uint8)
    # legenda simples
    bar = cv2.applyColorMap(np.linspace(255, 0, 150).astype(np.uint8)[:, None].repeat(18, 1), cv2.COLORMAP_JET)
    y0, x0 = 20, out.shape[1] - 50
    if out.shape[0] > 200 and out.shape[1] > 80:
        out[y0:y0 + 150, x0:x0 + 18] = bar
        cv2.putText(out, "max", (x0 - 8, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(out, "0", (x0 + 4, y0 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return out


def render_scanpath(stimulus: np.ndarray, fixations: list[dict], aois: list[dict]) -> np.ndarray:
    out = stimulus.copy()
    for a in aois:
        cv2.rectangle(out, (int(a["x"]), int(a["y"])), (int(a["x"] + a["w"]), int(a["y"] + a["h"])), (255, 120, 0), 2)
        cv2.putText(out, a["name"], (int(a["x"]) + 4, int(a["y"]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
    pts = [(int(f["stim_x"]), int(f["stim_y"])) for f in fixations]
    for p, q in zip(pts, pts[1:]):
        cv2.line(out, p, q, (0, 0, 200), 1, cv2.LINE_AA)
    for i, (f, p) in enumerate(zip(fixations, pts)):
        r = int(max(6, min(40, math.sqrt(f["duration_ms"]) * 1.2)))
        overlay = out.copy()
        cv2.circle(overlay, p, r, (0, 0, 220), -1, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.circle(out, p, r, (0, 0, 160), 1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (p[0] - 6, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------------
def in_aoi(f: dict, a: dict) -> bool:
    return a["x"] <= f["stim_x"] <= a["x"] + a["w"] and a["y"] <= f["stim_y"] <= a["y"] + a["h"]


def aoi_metrics(s: Session, aois: list[dict]) -> list[dict]:
    rows = []
    for a in aois:
        hits = [(i, f) for i, f in enumerate(s.fixations) if in_aoi(f, a)]
        visits, prev = 0, -2
        for i, _ in hits:
            if i != prev + 1:
                visits += 1
            prev = i
        rows.append({
            "participant": s.id,
            "aoi": a["name"],
            "hit": bool(hits),
            "ttff_ms": hits[0][1]["start_ms"] if hits else "",
            "fixation_count": len(hits),
            "dwell_time_ms": sum(f["duration_ms"] for _, f in hits),
            "visits": visits,
        })
    return rows


def session_summary(s: Session) -> dict:
    rec = s.meta.get("recording", {})
    val = s.meta.get("validation", {})
    cal = s.meta.get("calibration", {})
    durs = [f["duration_ms"] for f in s.fixations]
    row = {
        "participant": s.id,
        "status": s.meta.get("status"),
        "backend": s.meta.get("backend"),
        "calibration_model": cal.get("model"),
        "calib_train_rmse_px": cal.get("train_rmse_px"),
        "val_accuracy_px": val.get("accuracy_mean_px"),
        "val_accuracy_deg": val.get("accuracy_mean_deg", ""),
        "val_precision_rms_s2s_px": val.get("precision_rms_s2s_px"),
        "val_data_loss": val.get("data_loss"),
        "rec_duration_s": rec.get("duration_s"),
        "rec_fps": rec.get("fps"),
        "rec_data_loss": rec.get("data_loss"),
        "rec_gaze_on_stimulus": rec.get("gaze_on_stimulus_ratio"),
        "fixations": len(durs),
        "fixation_mean_ms": float(np.mean(durs)) if durs else "",
        "fixation_median_ms": float(np.median(durs)) if durs else "",
    }
    return row


def compare_models(s: Session, ppd: Optional[float]) -> list[dict]:
    """Treina cada modelo só com a calibração e avalia na validação (holdout)."""
    calib = [r for r in s.calib if r.get("phase") == "calibration" and r.get("valid") is True]
    valid = [r for r in s.calib if r.get("phase") == "validation"]
    if not calib or not valid:
        return []
    ids = [int(r["target_idx"]) for r in calib]
    feats = np.array([[r[n] for n in FEATURE_NAMES] for r in calib], float)
    tgts = np.array([[r["target_x"], r["target_y"]] for r in calib], float)
    feats, tgts, used = clean_calibration_samples(ids, feats, tgts)
    rows = []
    for model in CALIBRATION_MODELS:
        for use_head in (True, False):
            f = feats.copy()
            if not use_head:
                f[:, 2:] = 0.0  # zera características da cabeça (sd=0 -> coluna constante)
            try:
                cal = GazeCalibrator(model).fit(f, tgts)
            except (ValueError, np.linalg.LinAlgError):
                continue
            per_target: dict = {}
            for r in valid:
                key = (r["target_x"], r["target_y"])
                per_target.setdefault(key, [])
                if r.get("valid") is True:
                    x = np.array([r[n] for n in FEATURE_NAMES], float)
                    if not use_head:
                        x[2:] = 0.0
                    per_target[key].append(cal.predict(x))
            m = validation_metrics({k: np.array(v) for k, v in per_target.items()}, ppd)
            rows.append({"participant": s.id, "model": model, "head_features": use_head,
                         "calib_targets": used, "train_rmse_px": cal.train_rmse_px,
                         "val_accuracy_px": m["accuracy_mean_px"],
                         "val_accuracy_deg": m.get("accuracy_mean_deg", ""),
                         "val_precision_rms_s2s_px": m["precision_rms_s2s_px"]})
    return rows


def reprocess(s: Session, args, out_dir: Path) -> None:
    """Refaz o pipeline a partir de calibration_samples.csv e frames.csv."""
    meta = s.meta
    cfg = meta.get("config", {})
    blink_meta = meta.get("blink") or {}
    static_thr = blink_meta.get("static_threshold", meta.get("precheck", {}).get("blink_threshold", float("nan")))
    ratio = meta.get("precheck", {}).get("blink_ratio", 0.75)
    max_blink = args.max_blink_ms
    ppd = meta.get("screen", {}).get("px_per_degree")

    calib = [r for r in s.calib if r.get("phase") == "calibration"]
    val = [r for r in s.calib if r.get("phase") == "validation"]
    label_target_blinks(calib, static_thr, None, max_blink)
    ear_model = None
    if calib and "avg_ear" in calib[0]:
        try:
            ear_model = EarModel(ratio).fit([r["target_idx"] for r in calib], [r["eye_y"] for r in calib],
                                            [r["head_y"] for r in calib], [r["avg_ear"] for r in calib])
            label_target_blinks(calib, static_thr, ear_model, max_blink)
        except ValueError:
            ear_model = None
    label_target_blinks(val, static_thr, ear_model, max_blink)

    ok = [r for r in calib if r["valid"]]
    feats, tgts, used = clean_calibration_samples(
        [int(r["target_idx"]) for r in ok],
        np.array([[r[n] for n in FEATURE_NAMES] for r in ok], float),
        np.array([[r["target_x"], r["target_y"]] for r in ok], float))
    model = args.model or meta.get("calibration", {}).get("model", "poly2")
    ridge = args.ridge if args.ridge is not None else meta.get("calibration", {}).get("ridge", 1e-2)
    cal = GazeCalibrator(model, ridge).fit(feats, tgts)
    meta["calibration"] = {**cal.to_dict(), "targets_used": used}

    per_target: dict = {}
    for r in val:
        per_target.setdefault((r["target_x"], r["target_y"]), [])
        if r["valid"]:
            per_target[(r["target_x"], r["target_y"])].append(cal.predict([r[n] for n in FEATURE_NAMES]))
    vm = validation_metrics({k: np.array(v) for k, v in per_target.items()}, ppd)
    vm["data_loss"] = 1.0 - sum(r["valid"] for r in val) / len(val) if val else float("nan")
    meta["validation"] = vm

    lb_d = meta.get("recording", {}).get("letterbox")
    lb = Letterbox(**lb_d) if lb_d else Letterbox.fit(1920, 1080, 1920, 1080)
    params = ProcessingParams(
        filter_enabled=not cfg.get("no_filter", False),
        filter_min_cutoff=cfg.get("filter_min_cutoff", 0.8), filter_beta=cfg.get("filter_beta", 0.005),
        dispersion_px=args.dispersion_px or cfg.get("fixation_dispersion_px", 120.0),
        min_duration_ms=cfg.get("fixation_min_duration_ms", 100.0),
        max_gap_ms=cfg.get("fixation_max_gap_ms", 100.0), max_blink_ms=max_blink)
    rows, fix_rows, stats = process_recording(s.frames, cal, static_thr, ear_model, lb, params)
    stats["stop_reason"] = meta.get("recording", {}).get("stop_reason")
    meta["recording"] = stats
    meta["blink"] = {"static_threshold": static_thr, "max_blink_ms": max_blink,
                     "ear_model": ear_model.to_dict() if ear_model else None}
    meta["reprocessed"] = {"model": model, "ridge": ridge, "dispersion_px": params.dispersion_px}
    s.fixations = [f for f in fix_rows if math.isfinite(f["stim_x"]) and math.isfinite(f["stim_y"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "frames.csv", rows)
    write_csv(out_dir / "fixations.csv", fix_rows)
    write_csv(out_dir / "calibration_samples.csv", calib + val)
    print(f"[{s.id}] reprocessado: alvos de calibração={used} | validação {vm['accuracy_mean_px']:.0f}px"
          + (f" ({vm['accuracy_mean_deg']:.2f}°)" if "accuracy_mean_deg" in vm else "")
          + f" | perda na gravação {stats['data_loss'] * 100:.1f}% | {stats['fixations']} fixações")


# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Análise de sessões de eye tracking")
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--stimulus", type=Path)
    p.add_argument("--aois", type=Path)
    p.add_argument("--out", type=Path, default=Path("resultados"))
    p.add_argument("--sigma", type=float, default=40.0, help="desvio da gaussiana do heatmap (px do estímulo)")
    p.add_argument("--compare-models", action="store_true")
    p.add_argument("--reprocess", action="store_true", help="refaz o pipeline a partir dos dados brutos")
    p.add_argument("--model", choices=list(CALIBRATION_MODELS), help="(reprocess) modelo de calibração")
    p.add_argument("--ridge", type=float, help="(reprocess) regularização ridge")
    p.add_argument("--dispersion-px", type=float, help="(reprocess) limiar de dispersão do I-DT")
    p.add_argument("--max-blink-ms", type=float, default=500.0, help="(reprocess) duração máxima de piscada")
    args = p.parse_args(argv)

    sessions = [Session(d) for d in args.sessions]
    args.out.mkdir(parents=True, exist_ok=True)
    stimulus = load_stimulus(sessions, args.stimulus)
    aois = json.loads(args.aois.read_text(encoding="utf-8")) if args.aois else []
    h, w = stimulus.shape[:2]

    summaries, aoi_rows, model_rows, all_fix = [], [], [], []
    for s in sessions:
        if args.reprocess:
            reprocess(s, args, args.out / "reprocessed" / s.id)
        summaries.append(session_summary(s))
        aoi_rows += aoi_metrics(s, aois)
        all_fix += s.fixations
        cv2.imwrite(str(args.out / f"heatmap_{s.id}.png"),
                    render_heatmap(stimulus, density_map(s.fixations, (h, w), args.sigma)))
        cv2.imwrite(str(args.out / f"scanpath_{s.id}.png"), render_scanpath(stimulus, s.fixations, aois))
        if args.compare_models:
            model_rows += compare_models(s, s.meta.get("screen", {}).get("px_per_degree"))

    cv2.imwrite(str(args.out / "group_heatmap.png"),
                render_heatmap(stimulus, density_map(all_fix, (h, w), args.sigma)))
    write_csv(args.out / "summary.csv", summaries)
    write_csv(args.out / "aoi_metrics.csv", aoi_rows)
    write_csv(args.out / "model_comparison.csv", model_rows)

    acc = [r["val_accuracy_px"] for r in summaries if isinstance(r["val_accuracy_px"], (int, float))]
    loss = [r["rec_data_loss"] for r in summaries if isinstance(r["rec_data_loss"], (int, float))]
    print(f"{len(sessions)} sessão(ões) | fixações totais: {len(all_fix)}")
    if acc:
        print(f"acurácia de validação: média {np.mean(acc):.1f}px (DP {np.std(acc):.1f})")
    if loss:
        print(f"perda de dados na gravação: média {np.mean(loss) * 100:.1f}%")
    print(f"Resultados em {args.out.resolve()}")


if __name__ == "__main__":
    main()
