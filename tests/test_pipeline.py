"""Testes do núcleo algorítmico e da sessão completa com câmera/rosto simulados."""
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_session  # noqa: E402
import eyetracking_ux as et  # noqa: E402
from gaze_core import (GazeCalibrator, IDTFixationDetector, Letterbox, OneEuroFilter,  # noqa: E402
                       px_per_degree, robust_mask, validation_metrics)

W, H = 1600, 900


# --------------------------------------------------------------------------- unitários
def true_features(gx, gy, head=(0.0, 0.0), rng=None, noise=0.0):
    """Modelo sintético (não linear) de como a íris se move com o olhar."""
    u, v = gx / W - 0.5, gy / H - 0.5
    n = (lambda: rng.gauss(0, noise)) if rng else (lambda: 0.0)
    ex = 0.16 * u + 0.05 * u * u - 0.25 * head[0] + n()
    ey = 0.09 * v + 0.04 * v * v + 0.02 * u * v - 0.25 * head[1] + n()
    return [ex, ey, head[0] + n(), head[1] + n()]


def test_calibrator_recovers_nonlinear_mapping():
    rng = random.Random(0)
    feats, tgts = [], []
    for nx in np.linspace(0.1, 0.9, 3):
        for ny in np.linspace(0.1, 0.9, 3):
            for _ in range(30):
                head = (rng.gauss(0, 0.02), rng.gauss(0, 0.02))
                feats.append(true_features(nx * W, ny * H, head, rng, 0.001))
                tgts.append([nx * W, ny * H])
    lin = GazeCalibrator("linear").fit(np.array(feats), np.array(tgts))
    poly = GazeCalibrator("poly2").fit(np.array(feats), np.array(tgts))
    test = [(0.3 * W, 0.7 * H), (0.7 * W, 0.3 * H), (0.5 * W, 0.5 * H)]
    err = lambda c: np.mean([math.dist(c.predict(true_features(x, y)), (x, y)) for x, y in test])
    assert err(poly) < 30
    assert err(poly) < err(lin)
    clone = GazeCalibrator.from_dict(json.loads(json.dumps(poly.to_dict())))
    assert np.allclose(clone.predict(true_features(800, 450)), poly.predict(true_features(800, 450)))


def test_one_euro_reduces_jitter():
    rng = random.Random(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw, out = [], []
    for i in range(300):
        x = 500 + rng.gauss(0, 20)
        raw.append(x)
        out.append(f(x, i / 30))
    assert np.std(out[50:]) < 0.5 * np.std(raw[50:])


def test_idt_detects_fixations_and_ignores_gaps():
    det = IDTFixationDetector(dispersion_px=50, min_duration_ms=100, max_gap_ms=100)
    fixes, t = [], 0.0
    for cx, cy in [(100, 100), (600, 400), (300, 700)]:
        for i in range(15):  # 15 amostras a 33 ms ≈ 460 ms
            fx = det.add(t, cx + (i % 3), cy - (i % 2))
            if fx:
                fixes.append(fx)
            t += 33
        det.add(t, math.nan, math.nan)  # perda de dado não quebra
    last = det.flush()
    if last:
        fixes.append(last)
    assert len(fixes) == 3
    assert all(f.duration_ms >= 100 for f in fixes)
    assert math.dist((fixes[1].x, fixes[1].y), (600, 400)) < 5


def test_validation_metrics_and_degrees():
    pts = {(100.0, 100.0): np.array([[110, 100], [110, 100], [110, 100]]),
           (500.0, 500.0): np.array([[500, 520], [500, 520]])}
    m = validation_metrics(pts, px_per_deg=40.0)
    assert m["accuracy_mean_px"] == pytest.approx(15.0)
    assert m["accuracy_mean_deg"] == pytest.approx(15.0 / 40.0)
    assert m["precision_rms_s2s_px"] == pytest.approx(0.0)
    # 1920 px em 53 cm a 60 cm: ~37.9 px/grau
    assert px_per_degree(1920, 53.0, 60.0) == pytest.approx(37.9, abs=0.2)


def test_robust_mask_and_letterbox():
    vals = np.array([[0, 0]] * 10 + [[10, 10]], float) + np.random.default_rng(0).normal(0, 0.01, (11, 2))
    assert robust_mask(vals).tolist() == [True] * 10 + [False]
    lb = Letterbox.fit(1000, 1000, 1600, 900)
    assert lb.screen_to_stimulus(800, 450) == pytest.approx((500, 500))


# --------------------------------------------------------------------------- ponta a ponta
class FakeCap:
    def read(self):
        time.sleep(0.004)
        return True, np.zeros((480, 640, 3), np.uint8)


class FakeDisplay:
    def __init__(self):
        self.current_target = None
        self.frames = 0

    def open(self):
        pass

    def show(self, canvas, target=None):
        assert canvas.shape == (H, W, 3)
        self.current_target = target
        self.frames += 1

    def show_camera(self, frame):
        pass

    def key(self):
        return et.SPACE  # avança qualquer tela de instrução

    def close(self):
        pass


class FakeBackend:
    """Rosto simulado: segue o alvo exibido; na gravação percorre 3 regiões."""
    name = "fake"
    supports_blink = True

    def __init__(self, display):
        self.display = display
        self.rng = random.Random(3)
        self.blink_threshold = 0.2
        self.t0 = time.perf_counter()

    def process(self, frame):
        r = self.rng.random()
        if r < 0.03:
            return None  # perda de rastreamento
        blink = r < 0.05
        if self.display.current_target is not None:
            gx, gy = self.display.current_target
        else:
            k = int((time.perf_counter() - self.t0) / 0.5) % 3
            gx, gy = [(0.2 * W, 0.25 * H), (0.75 * W, 0.3 * H), (0.5 * W, 0.8 * H)][k]
        head = (self.rng.gauss(0, 0.01), self.rng.gauss(0, 0.01))
        f = true_features(gx, gy, head, self.rng, 0.002)
        ear = 0.1 if blink else 0.3 + self.rng.gauss(0, 0.01)
        return et.GazeSample(f[0], f[1], f[2], f[3], (320, 240), ear < self.blink_threshold, ear)

    def close(self):
        pass


def test_full_session_and_analysis(tmp_path):
    out = tmp_path / "P01"
    args = et.build_parser().parse_args([
        "--output-dir", str(out), "--participant", "P01", "--max-session-seconds", "2.5",
        "--target-seconds", "0.35", "--target-settle-seconds", "0.1", "--target-transition-seconds", "0.02",
        "--precheck-seconds", "0.4", "--screen-width-cm", "53", "--distance-cm", "60",
        "--fixation-dispersion-px", "120",
    ])
    display = FakeDisplay()
    backend = FakeBackend(display)
    import cv2
    stimulus = np.full((800, 1200, 3), 200, np.uint8)
    summary = et.run_session(args, et.Ctx(cv2, FakeCap(), backend, display, W, H, time.perf_counter()), stimulus)

    assert summary["status"] == "saved", summary.get("error")
    assert summary["validation"]["accuracy_mean_px"] < 60
    assert 0.0 < summary["recording"]["data_loss"] < 0.2
    assert summary["recording"]["fixations"] >= 3
    for name in ("session.json", "frames.csv", "fixations.csv", "calibration_samples.csv"):
        assert (out / name).exists()

    aois = tmp_path / "aois.json"
    aois.write_text(json.dumps([{"name": "topo_esq", "x": 0, "y": 0, "w": 500, "h": 400},
                                {"name": "rodape", "x": 300, "y": 550, "w": 600, "h": 250}]))
    stim_path = tmp_path / "stim.png"
    cv2.imwrite(str(stim_path), stimulus)
    res = tmp_path / "res"
    analyze_session.main([str(out), "--stimulus", str(stim_path), "--aois", str(aois),
                          "--out", str(res), "--compare-models"])
    for name in ("heatmap/group_heatmap.png", "heatmap/heatmap_P01.png", "scanpath/scanpath_P01.png",
                 "aois/aois.png", "aois/aoi_metrics.csv", "summary.csv", "model_comparison.csv"):
        assert (res / name).exists(), name
    aoi = analyze_session.read_csv(res / "aois" / "aoi_metrics.csv")
    assert all(r["hit"] for r in aoi)

    # mesma análise a partir de uma pasta de estudo
    study = tmp_path / "estudo"
    (study / "sessoes").mkdir(parents=True)
    shutil.copytree(out, study / "sessoes" / "P01")
    shutil.copy(aois, study / "aois.json")
    shutil.copy(stim_path, study / "estimulo.png")
    analyze_session.main(["--study", str(study)])
    for name in ("heatmap/heatmap_P01.png", "scanpath/scanpath_P01.png", "aois/aoi_metrics.csv"):
        assert (study / "resultados" / name).exists(), name
