"""Bitácora de pruebas y Deflated Sharpe (Bailey y López de Prado 2014).

Cada configuración corrida sobre la misma historia suma uno a N, incluidas las
descartadas y las variantes de prompt. El Sharpe que se exige para empatar con
el azar sube con N y baja con los años de datos.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TRIAL_LOG = ROOT / "data" / "eval" / "trial_log.jsonl"
EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, years: float) -> float:
    """Mejor Sharpe anual esperado entre n_trials estrategias sin habilidad.
    El Sharpe anual estimado con `years` años tiene desviación ~1/sqrt(years)."""
    if n_trials <= 1 or years <= 0:
        return 0.0
    sd = 1.0 / np.sqrt(years)
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sd * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(daily_returns: np.ndarray, n_trials: int, periods: int = 252) -> dict:
    """DSR = P(Sharpe real > el máximo esperado por azar entre n_trials).
    Corrige por asimetría y curtosis de los retornos. >0.95 es evidencia."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 20 or r.std() == 0:
        return {"sharpe_anual": None, "sr0_anual": None, "dsr": None, "n_obs": t}
    sr = r.mean() / r.std(ddof=1)
    sr0 = expected_max_sharpe(n_trials, t / periods) / np.sqrt(periods)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom)
    return {
        "sharpe_anual": round(float(sr * np.sqrt(periods)), 3),
        "sr0_anual": round(float(sr0 * np.sqrt(periods)), 3),
        "dsr": round(float(dsr), 3),
        "n_obs": t,
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
    }


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def log_trial(family: str, cfg: dict, result: dict, path: Path = TRIAL_LOG) -> int:
    """Agrega la prueba a la bitácora y devuelve cuántas configuraciones
    distintas lleva esa familia (el N del Deflated Sharpe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "hash": config_hash(cfg),
        "config": cfg,
        "result": result,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return n_trials(family, path)


def n_trials(family: str, path: Path = TRIAL_LOG) -> int:
    if not path.exists():
        return 0
    hashes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("family") == family:
            hashes.add(row.get("hash"))
    return len(hashes)
