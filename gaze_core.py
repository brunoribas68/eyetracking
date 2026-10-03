"""Núcleo algorítmico do rastreamento ocular (sem dependência de câmera/GUI).

Contém as peças que o artigo descreve na metodologia e que podem ser
testadas isoladamente:

- ``GazeCalibrator``: regressão (linear ou polinomial de 2º grau, com
  regularização ridge) que mapeia características do olho/cabeça para
  coordenadas da tela.
- ``OneEuroFilter`` / ``PointFilter``: suavização temporal adaptativa
  (Casiez et al., 2012).
- ``IDTFixationDetector``: detecção de fixações por dispersão (I-DT,
  Salvucci & Goldberg, 2000), em versão online.
- ``validation_metrics``: acurácia e precisão (Holmqvist et al., 2012).
- ``Letterbox``: conversão entre coordenadas da tela e do estímulo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

FEATURE_NAMES = ("eye_x", "eye_y", "head_x", "head_y")
CALIBRATION_MODELS = ("linear", "poly2")


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def is_finite_point(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y)


def robust_mask(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Máscara de amostras dentro de k desvios robustos (MAD) da mediana.

    ``values`` tem forma (N, D); uma amostra é descartada se qualquer
    dimensão for outlier.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    med = np.median(values, axis=0)
    mad = np.median(np.abs(values - med), axis=0) * 1.4826
    mad[mad < 1e-9] = np.inf  # dimensão constante: não descarta nada
    z = np.abs(values - med) / mad
    return np.all(z <= k, axis=1)


# ---------------------------------------------------------------------------
# calibração por regressão
# ---------------------------------------------------------------------------
def design_matrix(z: np.ndarray, model: str) -> np.ndarray:
    """Monta a matriz de projeto a partir das características padronizadas.

    linear: [1, ex, ey, hx, hy]
    poly2 : [1, ex, ey, ex*ey, ex^2, ey^2, hx, hy]
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    ex, ey, hx, hy = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    ones = np.ones_like(ex)
    if model == "linear":
        cols = [ones, ex, ey, hx, hy]
    elif model == "poly2":
        cols = [ones, ex, ey, ex * ey, ex ** 2, ey ** 2, hx, hy]
    else:
        raise ValueError(f"modelo de calibração desconhecido: {model}")
    return np.stack(cols, axis=1)


class GazeCalibrator:
    """Regressão ridge de características oculares -> coordenadas de tela (px)."""

    def __init__(self, model: str = "poly2", ridge: float = 1e-2) -> None:
        if model not in CALIBRATION_MODELS:
            raise ValueError(f"modelo deve ser um de {CALIBRATION_MODELS}")
        self.model = model
        self.ridge = float(ridge)
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.train_rmse_px: float = float("nan")
        self.n_samples: int = 0

    @property
    def fitted(self) -> bool:
        return self.weights is not None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "GazeCalibrator":
        f = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if f.ndim != 2 or f.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features deve ter forma (N, 4)")
        if y.shape != (f.shape[0], 2):
            raise ValueError("targets deve ter forma (N, 2)")
        self.mu = f.mean(axis=0)
        sd = f.std(axis=0)
        sd[sd < 1e-9] = 1.0
        self.sd = sd
        x = design_matrix((f - self.mu) / self.sd, self.model)
        if x.shape[0] < x.shape[1]:
            raise ValueError("amostras insuficientes para o modelo escolhido")
        penalty = self.ridge * np.eye(x.shape[1])
        penalty[0, 0] = 0.0  # não penaliza o intercepto
        self.weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        residual = x @ self.weights - y
        self.train_rmse_px = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        self.n_samples = int(f.shape[0])
        return self

    def predict_many(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrador não treinado")
        f = np.atleast_2d(np.asarray(features, dtype=float))
        return design_matrix((f - self.mu) / self.sd, self.model) @ self.weights

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        p = self.predict_many(np.asarray(features, dtype=float)[None, :])[0]
        return float(p[0]), float(p[1])

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "ridge": self.ridge,
            "feature_names": list(FEATURE_NAMES),
            "mu": None if self.mu is None else self.mu.tolist(),
            "sd": None if self.sd is None else self.sd.tolist(),
            "weights": None if self.weights is None else self.weights.tolist(),
            "train_rmse_px": self.train_rmse_px,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GazeCalibrator":
        c = cls(d["model"], d["ridge"])
        c.mu = np.asarray(d["mu"], dtype=float)
        c.sd = np.asarray(d["sd"], dtype=float)
        c.weights = np.asarray(d["weights"], dtype=float)
        c.train_rmse_px = float(d.get("train_rmse_px", float("nan")))
        c.n_samples = int(d.get("n_samples", 0))
        return c


def clean_calibration_samples(
    target_ids: Sequence[int],
    features: np.ndarray,
    targets: np.ndarray,
    min_samples_per_target: int = 5,
    k_mad: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove outliers por alvo e alvos com poucas amostras.

    Retorna (features, targets, n_alvos_utilizados).
    """
    ids = np.asarray(target_ids)
    f = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    keep_f, keep_y, used = [], [], 0
    for tid in np.unique(ids):
        sel = ids == tid
        ft, yt = f[sel], y[sel]
        if len(ft) < min_samples_per_target:
            continue
        mask = robust_mask(ft[:, :2], k_mad)  # outliers nas características do olho
        if mask.sum() < min_samples_per_target:
            continue
        keep_f.append(ft[mask])
        keep_y.append(yt[mask])
        used += 1
    if not keep_f:
        return np.empty((0, 4)), np.empty((0, 2)), 0
    return np.concatenate(keep_f), np.concatenate(keep_y), used


# ---------------------------------------------------------------------------
# filtro One Euro (Casiez, Roussel & Vogel, 2012)
# ---------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t_prev: Optional[float] = None
        self._x_prev = 0.0
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t_s: float) -> float:
        if self._t_prev is None:
            self._t_prev, self._x_prev, self._dx_prev = t_s, x, 0.0
            return x
        dt = t_s - self._t_prev
        if dt <= 0:
            return self._x_prev
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev
        self._t_prev, self._x_prev, self._dx_prev = t_s, x_hat, dx_hat
        return x_hat


class PointFilter:
    """Aplica One Euro em x e y e reinicia após lacunas (perda de dados)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, max_gap_s: float = 0.2,
                 enabled: bool = True) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)
        self.max_gap_s = max_gap_s
        self.enabled = enabled
        self._last_t: Optional[float] = None

    def __call__(self, x: float, y: float, t_s: float) -> tuple[float, float]:
        if not self.enabled:
            return x, y
        if self._last_t is not None and (t_s - self._last_t) > self.max_gap_s:
            self.fx.reset()
            self.fy.reset()
        self._last_t = t_s
        return self.fx(x, t_s), self.fy(y, t_s)


# ---------------------------------------------------------------------------
# detecção de fixações I-DT (Salvucci & Goldberg, 2000)
# ---------------------------------------------------------------------------
@dataclass
class Fixation:
    fixation_id: int
    start_ms: float
    end_ms: float
    duration_ms: float
    x: float
    y: float
    samples: int
    dispersion_px: float


def dispersion(points: Sequence[tuple[float, float, float]]) -> float:
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


class IDTFixationDetector:
    """Versão online do I-DT.

    Uma fixação é um trecho em que a dispersão (Δx + Δy) fica abaixo de
    ``dispersion_px`` por pelo menos ``min_duration_ms``. Amostras
    inválidas (NaN) não entram na janela; lacunas maiores que
    ``max_gap_ms`` encerram a janela atual.
    """

    def __init__(self, dispersion_px: float = 100.0, min_duration_ms: float = 100.0,
                 max_gap_ms: float = 100.0) -> None:
        self.dispersion_px = dispersion_px
        self.min_duration_ms = min_duration_ms
        self.max_gap_ms = max_gap_ms
        self.window: list[tuple[float, float, float]] = []
        self.next_id = 0

    def _window_duration(self) -> float:
        if len(self.window) < 2:
            return 0.0
        return self.window[-1][0] - self.window[0][0]

    @property
    def current_fixation_id(self) -> int:
        """ID provisório se a janela atual já qualifica como fixação; senão -1."""
        return self.next_id if self._window_duration() >= self.min_duration_ms else -1

    def _close(self) -> Optional[Fixation]:
        fix = None
        if self.window and self._window_duration() >= self.min_duration_ms:
            xs = [p[1] for p in self.window]
            ys = [p[2] for p in self.window]
            fix = Fixation(
                fixation_id=self.next_id,
                start_ms=self.window[0][0],
                end_ms=self.window[-1][0],
                duration_ms=self._window_duration(),
                x=float(sum(xs) / len(xs)),
                y=float(sum(ys) / len(ys)),
                samples=len(self.window),
                dispersion_px=dispersion(self.window),
            )
            self.next_id += 1
        self.window = []
        return fix

    def add(self, t_ms: float, x: float, y: float) -> Optional[Fixation]:
        if not is_finite_point(x, y):
            return None
        p = (t_ms, x, y)
        finalized = None
        if self.window and (t_ms - self.window[-1][0]) > self.max_gap_ms:
            finalized = self._close()
            self.window = [p]
            return finalized
        candidate = self.window + [p]
        if dispersion(candidate) <= self.dispersion_px:
            self.window = candidate
            return None
        if self._window_duration() >= self.min_duration_ms:
            finalized = self._close()
            self.window = [p]
            return finalized
        # janela ainda não é fixação: desliza descartando as amostras mais antigas
        while self.window and dispersion(self.window + [p]) > self.dispersion_px:
            self.window.pop(0)
        self.window.append(p)
        return None

    def flush(self) -> Optional[Fixation]:
        return self._close()


# ---------------------------------------------------------------------------
# qualidade dos dados: acurácia e precisão (Holmqvist, Nyström & Mulvey, 2012)
# ---------------------------------------------------------------------------
def px_per_degree(screen_w_px: float, screen_w_cm: float, distance_cm: float) -> float:
    """Pixels correspondentes a 1° de ângulo visual no centro da tela."""
    cm_per_degree = 2.0 * distance_cm * math.tan(math.radians(0.5))
    return (screen_w_px / screen_w_cm) * cm_per_degree


def rms_s2s(points: np.ndarray) -> float:
    """Precisão RMS entre amostras sucessivas."""
    if len(points) < 2:
        return float("nan")
    d = np.diff(points, axis=0)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def validation_metrics(
    per_target: dict[tuple[float, float], np.ndarray],
    px_per_deg: Optional[float] = None,
    expected_samples: Optional[int] = None,
) -> dict:
    """Calcula acurácia (offset médio) e precisão por alvo e no geral.

    ``per_target`` mapeia (alvo_x, alvo_y) -> array (N, 2) de estimativas.
    """
    rows = []
    for (tx, ty), pts in per_target.items():
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) == 0:
            rows.append({"target_x": tx, "target_y": ty, "n": 0, "offset_px": float("nan"),
                         "rms_s2s_px": float("nan"), "sd_px": float("nan")})
            continue
        mean = pts.mean(axis=0)
        offset = float(math.dist((tx, ty), mean))
        sd = float(np.sqrt(np.sum(pts.var(axis=0))))
        rows.append({"target_x": tx, "target_y": ty, "n": int(len(pts)), "offset_px": offset,
                     "rms_s2s_px": rms_s2s(pts), "sd_px": sd})

    offsets = [r["offset_px"] for r in rows if math.isfinite(r["offset_px"])]
    s2s = [r["rms_s2s_px"] for r in rows if math.isfinite(r["rms_s2s_px"])]
    sds = [r["sd_px"] for r in rows if math.isfinite(r["sd_px"])]
    summary = {
        "n_targets": len(rows),
        "n_targets_with_data": len(offsets),
        "accuracy_mean_px": float(np.mean(offsets)) if offsets else float("nan"),
        "accuracy_median_px": float(np.median(offsets)) if offsets else float("nan"),
        "accuracy_max_px": float(np.max(offsets)) if offsets else float("nan"),
        "precision_rms_s2s_px": float(np.mean(s2s)) if s2s else float("nan"),
        "precision_sd_px": float(np.mean(sds)) if sds else float("nan"),
        "per_target": rows,
    }
    if px_per_deg:
        for key in ("accuracy_mean", "accuracy_median", "accuracy_max", "precision_rms_s2s", "precision_sd"):
            summary[key + "_deg"] = summary[key + "_px"] / px_per_deg
    return summary


# ---------------------------------------------------------------------------
# estímulo ajustado à tela (letterbox)
# ---------------------------------------------------------------------------
@dataclass
class Letterbox:
    scale: float
    off_x: float
    off_y: float
    stim_w: int
    stim_h: int

    @classmethod
    def fit(cls, stim_w: int, stim_h: int, screen_w: int, screen_h: int) -> "Letterbox":
        scale = min(screen_w / stim_w, screen_h / stim_h)
        new_w, new_h = stim_w * scale, stim_h * scale
        return cls(scale, (screen_w - new_w) / 2.0, (screen_h - new_h) / 2.0, stim_w, stim_h)

    def screen_to_stimulus(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.off_x) / self.scale, (y - self.off_y) / self.scale

    def inside_stimulus(self, sx: float, sy: float) -> bool:
        return is_finite_point(sx, sy) and 0 <= sx < self.stim_w and 0 <= sy < self.stim_h

    def to_dict(self) -> dict:
        return {"scale": self.scale, "off_x": self.off_x, "off_y": self.off_y,
                "stim_w": self.stim_w, "stim_h": self.stim_h}


def grid_points(levels: Iterable[float]) -> list[tuple[float, float]]:
    levels = list(levels)
    return [(x, y) for y in levels for x in levels]


# ---------------------------------------------------------------------------
# piscadas: limiar dependente do olhar + regra de duração
# ---------------------------------------------------------------------------
def unflag_long_episodes(t_ms: Sequence[float], low: Sequence[bool], max_blink_ms: float = 500.0,
                         max_gap_ms: float = 100.0) -> np.ndarray:
    """Mantém como piscada só os episódios curtos de EAR baixa.

    Uma piscada dura ~100-400 ms. Episódios mais longos são, em geral, o
    olhar dirigido para baixo (a pálpebra superior acompanha o olho) e não
    devem ser descartados.
    """
    t = np.asarray(t_ms, dtype=float)
    low = np.asarray(low, dtype=bool)
    out = low.copy()
    i, n = 0, len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1] and (t[j + 1] - t[j]) <= max_gap_ms:
            j += 1
        if t[j] - t[i] > max_blink_ms:
            out[i:j + 1] = False
        i = j + 1
    return out


class EarModel:
    """EAR esperada com olhos abertos em função do olhar vertical e da cabeça.

    Ajustada com as medianas por alvo da calibração (a mediana é robusta às
    piscadas). Piscada = EAR < ratio * EAR esperada.
    """

    def __init__(self, ratio: float = 0.75) -> None:
        self.ratio = ratio
        self.coef: Optional[np.ndarray] = None

    def fit(self, target_ids, eye_y, head_y, ear) -> "EarModel":
        ids = np.asarray(target_ids)
        ey, hy, e = (np.asarray(v, dtype=float) for v in (eye_y, head_y, ear))
        ok = np.isfinite(ey) & np.isfinite(hy) & np.isfinite(e)
        rows = []
        for tid in np.unique(ids[ok]):
            sel = ok & (ids == tid)
            if sel.sum() >= 5:
                rows.append((np.median(ey[sel]), np.median(hy[sel]), np.median(e[sel])))
        if len(rows) < 4:
            raise ValueError("alvos insuficientes para o modelo de EAR")
        a = np.array(rows)
        x = np.column_stack([np.ones(len(a)), a[:, 0], a[:, 1]])
        penalty = 1e-6 * np.eye(3)
        penalty[0, 0] = 0
        self.coef = np.linalg.solve(x.T @ x + penalty, x.T @ a[:, 2])
        return self

    def expected(self, eye_y, head_y) -> np.ndarray:
        return self.coef[0] + self.coef[1] * np.asarray(eye_y, float) + self.coef[2] * np.asarray(head_y, float)

    def thresholds(self, eye_y, head_y) -> np.ndarray:
        return self.ratio * self.expected(eye_y, head_y)

    def to_dict(self) -> dict:
        return {"ratio": self.ratio, "coef": None if self.coef is None else self.coef.tolist(),
                "terms": ["1", "eye_y", "head_y"]}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["EarModel"]:
        if not d or d.get("coef") is None:
            return None
        m = cls(d["ratio"])
        m.coef = np.asarray(d["coef"], float)
        return m


def low_ear_mask(ear, eye_y, head_y, static_threshold: float, ear_model: Optional[EarModel]) -> np.ndarray:
    ear = np.asarray(ear, dtype=float)
    if ear_model is not None:
        thr = ear_model.thresholds(eye_y, head_y)
    else:
        thr = np.full(ear.shape, static_threshold if static_threshold is not None else np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(ear) & np.isfinite(thr) & (ear < thr)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def label_target_blinks(records: list[dict], static_threshold: float, ear_model: Optional[EarModel],
                        max_blink_ms: float) -> None:
    """Marca blink/valid nas amostras de calibração/validação (in place), por alvo."""
    groups: dict = {}
    for r in records:
        groups.setdefault((r["phase"], r["target_idx"]), []).append(r)
    for recs in groups.values():
        t = [_f(r["timestamp_ms"]) for r in recs]
        if all("avg_ear" in r and r["avg_ear"] not in ("", None) for r in recs):
            low = low_ear_mask([_f(r["avg_ear"]) for r in recs], [_f(r["eye_y"]) for r in recs],
                               [_f(r["head_y"]) for r in recs], static_threshold, ear_model)
        else:  # sessões antigas sem EAR salva: usa a marcação original
            low = np.array([r.get("blink") in (True, "True") for r in recs])
        blink = unflag_long_episodes(t, low, max_blink_ms)
        for r, b in zip(recs, blink):
            face = math.isfinite(_f(r["eye_x"]))
            r["blink"] = bool(b)
            r["valid"] = bool(face and not b)


@dataclass
class ProcessingParams:
    filter_enabled: bool = True
    filter_min_cutoff: float = 0.8
    filter_beta: float = 0.005
    dispersion_px: float = 120.0
    min_duration_ms: float = 100.0
    max_gap_ms: float = 100.0
    max_blink_ms: float = 500.0


def process_recording(raw_rows: list[dict], cal: GazeCalibrator, static_threshold: float,
                      ear_model: Optional[EarModel], lb: Letterbox,
                      p: ProcessingParams) -> tuple[list[dict], list[dict], dict]:
    """Pós-processa a gravação: piscadas -> olhar -> filtro -> fixações (I-DT).

    Feito depois da coleta para poder usar a duração dos episódios de EAR
    baixa e para permitir reprocessar sessões com outros parâmetros.
    """
    t = np.array([_f(r["timestamp_ms"]) for r in raw_rows])
    feats = np.array([[_f(r[n]) for n in FEATURE_NAMES] for r in raw_rows]).reshape(-1, 4)
    ear = np.array([_f(r.get("avg_ear")) for r in raw_rows])
    face = np.all(np.isfinite(feats), axis=1)
    low = low_ear_mask(ear, feats[:, 1], feats[:, 3], static_threshold, ear_model)
    blink = unflag_long_episodes(t, low, p.max_blink_ms)
    valid = face & ~blink

    filt = PointFilter(p.filter_min_cutoff, p.filter_beta, enabled=p.filter_enabled)
    idt = IDTFixationDetector(p.dispersion_px, p.min_duration_ms, p.max_gap_ms)
    fixations: list[Fixation] = []
    rows = []
    for i, r in enumerate(raw_rows):
        raw = (float("nan"), float("nan"))
        sm = raw
        if valid[i]:
            raw = cal.predict(feats[i])
            sm = filt(raw[0], raw[1], t[i] / 1000.0)
        fx = idt.add(t[i], *sm)
        if fx:
            fixations.append(fx)
        sx, sy = lb.screen_to_stimulus(*sm)
        rows.append({
            "frame_idx": int(_f(r.get("frame_idx", i))), "timestamp_ms": t[i],
            "face": bool(face[i]), "blink": bool(blink[i]), "valid": bool(valid[i]),
            **{n: feats[i, k] for k, n in enumerate(FEATURE_NAMES)}, "avg_ear": ear[i],
            "gaze_raw_x": raw[0], "gaze_raw_y": raw[1], "gaze_x": sm[0], "gaze_y": sm[1],
            "stim_x": sx, "stim_y": sy, "fixation_id": -1,
        })
    last = idt.flush()
    if last:
        fixations.append(last)

    fix_rows = []
    for fx in fixations:
        sx, sy = lb.screen_to_stimulus(fx.x, fx.y)
        fix_rows.append({"fixation_id": fx.fixation_id, "start_ms": fx.start_ms, "end_ms": fx.end_ms,
                         "duration_ms": fx.duration_ms, "x": fx.x, "y": fx.y, "stim_x": sx, "stim_y": sy,
                         "samples": fx.samples, "dispersion_px": fx.dispersion_px})
        for row in rows:
            if fx.start_ms <= row["timestamp_ms"] <= fx.end_ms and row["valid"]:
                row["fixation_id"] = fx.fixation_id

    n = len(rows)
    n_valid = int(valid.sum())
    duration_s = (t[-1] - t[0]) / 1000.0 if n > 1 else 0.0
    on_stim = sum(1 for r in rows if r["valid"] and lb.inside_stimulus(r["stim_x"], r["stim_y"]))
    stats = {
        "duration_s": duration_s,
        "frames": n,
        "fps": n / duration_s if duration_s > 0 else float("nan"),
        "face_loss": 1.0 - face.mean() if n else float("nan"),
        "blink_frames": int(blink.sum()),
        "long_low_ear_frames_kept": int((low & ~blink).sum()),
        "data_loss": 1.0 - n_valid / n if n else float("nan"),
        "gaze_on_stimulus_ratio": on_stim / n_valid if n_valid else float("nan"),
        "fixations": len(fix_rows),
        "letterbox": lb.to_dict(),
    }
    return rows, fix_rows, stats
