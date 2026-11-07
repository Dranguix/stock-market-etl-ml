"""
Tasas de mercado de dinero mexicanas como features para el modelo.

Fuente: DataBursátil /v2/tasas (snapshot diario)
Almacén: data/rates/tasas/YYYY-MM-DD.json

Features computadas:
    tiie_spread      — TIIE91 − TIIE28: pendiente corta interbank
    cete_slope       — CETE364 − CETE28: pendiente de la curva de CETES (proxy 1y vs 1m)
    tiie28_level     — nivel absoluto TIIE28 (% anualizado)
    tiie28_ret5      — cambio de TIIE28 en 5 días hábiles (bp)
    tasa_obj_delta   — cambio de Tasa_Objetivo en 63 días (hz de política monetaria)

Todas son market-wide (iguales para todos los tickers en una fecha dada).
Se añaden a un ticker DataFrame via add_rates_features().

Uso:
    from src.ml.rates_features import (
        fetch_tasas_snapshot, build_rates_panel,
        compute_rates_features, add_rates_features,
        RATES_FEATURE_COLS,
    )
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
RATES_DIR = _ROOT / "data" / "rates" / "tasas"

# ── Columnas de salida ─────────────────────────────────────────────────────────
RATES_FEATURE_COLS: list[str] = [
    "tiie_spread",      # TIIE91 - TIIE28 (bp)
    "cete_slope",       # CETE364 - CETE28 (bp)
    "tiie28_level",     # TIIE28 (% anualizado)
    "tiie28_ret5",      # cambio TIIE28 últimos 5d hábiles (bp)
    "tasa_obj_delta",   # cambio Tasa_Objetivo 63d (bp)
]

# ── Mapeo de claves DataBursátil → nombres internos ───────────────────────────
_KEY_MAP = {
    "TIIE28":        "tiie28",
    "TIIE91":        "tiie91",
    "TIIE182":       "tiie182",
    "CETE28":        "cete28",
    "CETE 91":       "cete91",   # nota: hay un espacio en la respuesta de la API
    "CETE91":        "cete91",
    "CETE182":       "cete182",
    "CETE364":       "cete364",
    "Tasa_Objetivo": "tasa_obj",
}


# ── 1. Snapshot diario ─────────────────────────────────────────────────────────

def fetch_tasas_snapshot(
    token: str | None = None,
    fecha: date | str | None = None,
    *,
    timeout: int = 15,
    sleep: float = 0.3,
) -> dict[str, float] | None:
    """
    Llama a /v2/tasas y guarda el snapshot en RATES_DIR/YYYY-MM-DD.json.

    Parameters
    ----------
    token : token DataBursátil (default: var env DATABURSATIL_API_TOKEN)
    fecha : fecha del snapshot (default: hoy)
    timeout : segundos para la request
    sleep   : pausa antes de la request (para cron)

    Returns
    -------
    dict con claves normalizadas (tiie28, tiie91, cete28, etc.) o None si falla.
    """
    if token is None:
        token = os.environ.get("DATABURSATIL_API_TOKEN", "")
    if not token:
        logger.error("DATABURSATIL_API_TOKEN no definido")
        return None

    if fecha is None:
        fecha = date.today()
    fecha_str = str(fecha)[:10]

    out_file = RATES_DIR / f"{fecha_str}.json"
    if out_file.exists():
        logger.debug("Snapshot tasas %s ya existe, saltando fetch", fecha_str)
        return json.loads(out_file.read_text())

    import requests

    if sleep > 0:
        time.sleep(sleep)

    try:
        r = requests.get(
            "https://api.databursatil.com/v2/tasas",
            params={"token": token},
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        logger.error("Error fetching tasas: %s", exc)
        return None

    # Normalizar claves
    snapshot: dict[str, float] = {}
    for api_key, internal_key in _KEY_MAP.items():
        if api_key in raw and isinstance(raw[api_key], dict):
            t = raw[api_key].get("t")
            if t is not None:
                snapshot[internal_key] = float(t)

    if not snapshot:
        logger.warning("Snapshot tasas vacío para %s", fecha_str)
        return None

    RATES_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    logger.info("Snapshot tasas %s guardado (%d tasas)", fecha_str, len(snapshot))
    return snapshot


# ── 2. Panel histórico ─────────────────────────────────────────────────────────

def build_rates_panel(rates_dir: Path | None = None) -> pd.DataFrame:
    """
    Lee todos los snapshots en RATES_DIR y devuelve un DataFrame:
        index: pd.DatetimeIndex (fecha hábil)
        cols : tiie28, tiie91, cete28, cete91, cete182, cete364, tasa_obj

    Los valores son tasas anualizadas en %.
    """
    rates_dir = rates_dir or RATES_DIR
    if not rates_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for f in sorted(rates_dir.glob("????-??-??.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        data["fecha"] = f.stem
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("fecha")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Convertir a float, forzar numeric
    for col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    return panel


# ── 3. Features ───────────────────────────────────────────────────────────────

def compute_rates_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Añade RATES_FEATURE_COLS al panel de tasas.

    Input : DataFrame de build_rates_panel (fechas × tasas crudas)
    Output: mismo DataFrame con las 5 columnas nuevas añadidas.
    """
    p = panel.copy()

    # Level features (solo necesitan el snapshot de hoy)
    if "tiie91" in p.columns and "tiie28" in p.columns:
        p["tiie_spread"] = p["tiie91"] - p["tiie28"]
    else:
        p["tiie_spread"] = np.nan

    if "cete364" in p.columns and "cete28" in p.columns:
        p["cete_slope"] = p["cete364"] - p["cete28"]
    else:
        p["cete_slope"] = np.nan

    p["tiie28_level"] = p.get("tiie28", pd.Series(dtype=float))

    # Rolling features (necesitan historial — NaN al inicio)
    if "tiie28" in p.columns:
        p["tiie28_ret5"] = p["tiie28"].diff(5)
    else:
        p["tiie28_ret5"] = np.nan

    if "tasa_obj" in p.columns:
        p["tasa_obj_delta"] = p["tasa_obj"].diff(63)
    else:
        p["tasa_obj_delta"] = np.nan

    return p[RATES_FEATURE_COLS]


# ── 4. Merge en DataFrame de ticker ──────────────────────────────────────────

def add_rates_features(
    feat_df: pd.DataFrame,
    rates_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Añade RATES_FEATURE_COLS a un DataFrame de features de un ticker.

    Parameters
    ----------
    feat_df      : DataFrame con DatetimeIndex (fechas hábiles)
    rates_panel  : DataFrame de compute_rates_features. Si None, lo construye.

    Returns
    -------
    feat_df con las columnas RATES_FEATURE_COLS añadidas (forward-fill para
    días en que no hay snapshot).
    """
    if rates_panel is None or rates_panel.empty:
        try:
            raw = build_rates_panel()
            if raw.empty:
                for col in RATES_FEATURE_COLS:
                    feat_df[col] = np.nan
                return feat_df
            rates_panel = compute_rates_features(raw)
        except Exception as exc:
            logger.warning("No se pudo construir rates_panel: %s", exc)
            for col in RATES_FEATURE_COLS:
                feat_df[col] = np.nan
            return feat_df

    idx = feat_df.index if hasattr(feat_df.index, "freq") else pd.DatetimeIndex(feat_df.index)
    rates_reindexed = rates_panel.reindex(idx, method="ffill")

    for col in RATES_FEATURE_COLS:
        if col in rates_reindexed.columns:
            feat_df[col] = rates_reindexed[col].values
        else:
            feat_df[col] = np.nan

    return feat_df


# ── 5. Bootstrap de historial vía Banxico ────────────────────────────────────

_BANXICO_SERIES = {
    "tiie28":   "CF102",
    "tiie91":   "CF110",
    "cete28":   "SF43783",
    "cete364":  "SF43786",
    "tasa_obj": "SF61745",
}


def backfill_from_banxico(
    start: str = "2024-01-01",
    end: str | None = None,
    bmx_token: str | None = None,
    rates_dir: Path | None = None,
    *,
    sleep: float = 0.5,
) -> int:
    """
    Bootstrapea el historial de tasas usando la API pública de Banxico.

    Requiere un token Banxico (gratis en https://www.banxico.org.mx/SieAPIRest/).
    Si no se provee, intenta sin token (limitado).

    Returns
    -------
    Número de días guardados.
    """
    import requests

    rates_dir = rates_dir or RATES_DIR
    rates_dir.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    headers = {}
    if bmx_token:
        headers["Bmx-Token"] = bmx_token

    base = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"

    # Fetch cada serie
    frames: dict[str, pd.Series] = {}
    for col, serie in _BANXICO_SERIES.items():
        url = f"{base}/{serie}/datos/{start}/{end}"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            obs = data["bmx"]["series"][0].get("datos", [])
            s = pd.Series(
                {row["fecha"]: float(row["dato"].replace(",", "")) if row["dato"] != "N/E" else np.nan
                 for row in obs}
            )
            s.index = pd.to_datetime(s.index, format="%d/%m/%Y", errors="coerce")
            s = s.dropna(how="all")
            frames[col] = s
            logger.info("Banxico: %s — %d observaciones", col, len(s))
            time.sleep(sleep)
        except Exception as exc:
            logger.warning("Banxico serie %s falló: %s", col, exc)

    if not frames:
        logger.error("Banxico bootstrap falló — sin datos")
        return 0

    # Unir por fecha
    panel = pd.DataFrame(frames).sort_index()
    saved = 0
    for fecha, row in panel.iterrows():
        fecha_str = fecha.strftime("%Y-%m-%d")
        out_file = rates_dir / f"{fecha_str}.json"
        if out_file.exists():
            continue  # no sobreescribir snapshots reales
        snapshot = row.dropna().to_dict()
        if snapshot:
            out_file.write_text(json.dumps(snapshot, indent=2))
            saved += 1

    logger.info("Banxico bootstrap: %d días guardados", saved)
    return saved
