"""Un reporte 360° de la BMV: una fila completa por emisora, con un score
multifactor (momentum + calidad + baja volatilidad + valor) calculado en código,
no adivinado por el LLM. Cabe en una sola llamada."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "data" / "market"
NEWS = ROOT / "data" / "news"
FUNDAMENTALS = ROOT / "data" / "fundamentals.parquet"
DIVIDENDS = ROOT / "data" / "dividends" / "dividends.json"
REPORTED_EPS = ROOT / "data" / "eps" / "reported_eps.json"


def _bmv_symbols() -> dict[str, str]:
    from src.data.universes import UNIVERSES
    for cfg in UNIVERSES.values():
        if cfg.get("slug") == "bmv":
            return dict(cfg["tickers"])
    return {}


def _bars(symbol: str) -> pd.DataFrame | None:
    path = MARKET / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def _news_by_ticker(day: str) -> dict[str, dict]:
    path = NEWS / f"bmv_{day}.json"
    if not path.exists():
        path = NEWS / "latest.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {r["ticker"]: r for r in rows if r.get("ticker")}


def _pct(now: float, then: float) -> str:
    if then is None or then <= 0:
        return "na"
    return f"{(now / then - 1) * 100:+.1f}%"


def _load_fundamentals() -> pd.DataFrame | None:
    if not FUNDAMENTALS.exists():
        return None
    df = pd.read_parquet(FUNDAMENTALS)
    df["available_date"] = pd.to_datetime(df["available_date"]).dt.tz_localize(None)
    return df


def _load_dividends() -> dict:
    if not DIVIDENDS.exists():
        return {}
    return json.loads(DIVIDENDS.read_text())


def _load_reported_eps() -> dict:
    if not REPORTED_EPS.exists():
        return {}
    return json.loads(REPORTED_EPS.read_text())


def _fundamentals_asof(fund_df: pd.DataFrame | None, name: str, asof: pd.Timestamp) -> dict:
    if fund_df is None:
        return {}
    rows = fund_df[(fund_df["ticker"] == name) & (fund_df["available_date"] <= asof)]
    if rows.empty:
        return {}
    r = rows.sort_values("available_date").iloc[-1]
    return {
        "gross_margin": float(r["fund_gross_margin"]) if pd.notna(r["fund_gross_margin"]) else None,
        "oper_margin": float(r["fund_oper_margin"]) if pd.notna(r["fund_oper_margin"]) else None,
        "net_margin": float(r["fund_net_margin"]) if pd.notna(r["fund_net_margin"]) else None,
        "revenue_yoy": float(r["fund_revenue_yoy"]) if pd.notna(r["fund_revenue_yoy"]) else None,
        "earnings_yoy": float(r["fund_earnings_yoy"]) if pd.notna(r["fund_earnings_yoy"]) else None,
    }


def _div_yield_ttm(name: str, close: float, asof: pd.Timestamp, divs_all: dict) -> float | None:
    rows = divs_all.get(name) or []
    cutoff = asof - pd.Timedelta(days=365)
    total = 0.0
    seen = set()
    for item in rows:
        ex = item.get("ex_date") or item.get("pago_date")
        pago = item.get("pago")
        if not ex or pago is None:
            continue
        ts = pd.Timestamp(ex)
        if ts in seen or ts > asof or ts < cutoff:
            continue
        seen.add(ts)
        total += float(pago)
    if close <= 0:
        return None
    return total / close


def _earnings_yield_ttm(symbol: str, close: float, asof: pd.Timestamp, eps_all: dict) -> float | None:
    key = symbol if symbol in eps_all else f"{symbol}.MX"
    rows = eps_all.get(key) or eps_all.get(symbol) or []
    if not rows:
        return None
    usable = [r for r in rows if pd.Timestamp(r["announced"]) + pd.Timedelta(days=1) <= asof]
    if len(usable) < 4:
        return None
    usable = sorted(usable, key=lambda r: r["announced"])[-4:]
    ttm = sum(float(r["reported"]) for r in usable)
    if close <= 0:
        return None
    return ttm / close


def _macro_context(asof: pd.Timestamp) -> dict:
    out: dict[str, str] = {}
    specs = {
        "USD/MXN": MARKET / "MXN=X.parquet",
        "WTI": MARKET / "CL=F.parquet",
        "S&P500": MARKET / "GSPC.parquet",
        "IPC (NAFTRAC)": MARKET / "NAFTRAC.MX.parquet",
    }
    for label, path in specs.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        hist = df.loc[:asof]
        if len(hist) > 5:
            out[label] = _pct(float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[-6]))
    return out


def _regime(asof: pd.Timestamp) -> str:
    try:
        from src.ml.regime import RegimeDetector
        naf = _bars_raw(MARKET / "NAFTRAC.MX.parquet")
        sp = _bars_raw(MARKET / "GSPC.parquet")
        if naf is None or len(naf) < 100:
            return "sin dato"
        det = RegimeDetector().fit(naf.loc[:asof], sp.loc[:asof] if sp is not None else None)
        proba = det.state_proba(naf.loc[:asof], sp.loc[:asof] if sp is not None else None)
        label = max(proba, key=proba.get)
        pretty = ", ".join(f"{k} {v*100:.0f}%" for k, v in sorted(proba.items(), key=lambda x: -x[1]))
        return f"{label} ({pretty})"
    except Exception:
        return "sin dato"


def _bars_raw(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def atr_stop(hist: pd.DataFrame, close: float, mult: float = 1.5, window: int = 14) -> float:
    """Stop de referencia a mult*ATR(window). mult=1.5/window=14 es para trading de
    días; para sostener semanas/un mes usa un mult más ancho (2.5-3.5) para no cortarte
    con el ruido normal del periodo."""
    stop = close * 0.98
    if len(hist) >= window + 1 and "High" in hist.columns:
        span = (hist["High"] - hist["Low"]).tail(window).mean()
        if span and span > 0:
            stop = close - mult * float(span)
    return stop


def _zscore(values: dict[str, float | None]) -> dict[str, float]:
    xs = np.array([v for v in values.values() if v is not None], dtype=float)
    if len(xs) < 3 or xs.std() == 0:
        return {k: 0.0 for k in values}
    mu, sd = xs.mean(), xs.std()
    return {k: (0.0 if v is None else float((v - mu) / sd)) for k, v in values.items()}


def _composite_scores(metrics: dict[str, dict]) -> dict[str, float]:
    """Score multifactor estilo quant: momentum + calidad + baja volatilidad + valor.
    Cada pata se vuelve z-score entre las emisoras del día para que sean comparables."""
    mom = {t: m.get("year_raw") for t, m in metrics.items()}
    quality = {
        t: (m.get("fund", {}) or {}).get("oper_margin")
        for t, m in metrics.items()
    }
    lowvol = {t: (None if m.get("vol20") is None else -m["vol20"]) for t, m in metrics.items()}
    value = {t: m.get("eps_y") for t, m in metrics.items()}

    z_mom, z_qual, z_vol, z_val = _zscore(mom), _zscore(quality), _zscore(lowvol), _zscore(value)
    out = {}
    for t in metrics:
        out[t] = round(
            0.40 * z_mom.get(t, 0.0)
            + 0.25 * z_qual.get(t, 0.0)
            + 0.20 * z_vol.get(t, 0.0)
            + 0.15 * z_val.get(t, 0.0),
            2,
        )
    return out


def _all_symbols() -> dict[str, str]:
    """Todos los universos juntos: BMV + USA + ETFs + cripto + FIBRAs."""
    from src.data.universes import UNIVERSES
    out: dict[str, str] = {}
    for cfg in UNIVERSES.values():
        out.update(cfg["tickers"])
    return out


def build_bmv_report(
    asof: pd.Timestamp | None = None,
    held: dict[str, int] | None = None,
    cost_note: bool = False,
    exclude: set[str] | None = None,
    stop_mult: float = 2.5,
    universe: str = "bmv",
    index_options: tuple[tuple[str, str], ...] = (("NAFTRAC", "NAFTRAC.MX"),),
) -> str:
    """held: ticker -> días que el libro lleva sostenido esa posición (para frenar el
    overtrading). cost_note: si True, agrega una advertencia explícita sobre el costo
    de ida y vuelta por rotar de nombre. exclude: tickers en "enfriamiento" (p.ej.
    tocaron su stop hace poco) que no se ofrecen este periodo. universe: "bmv" (35
    emisoras MX) o "full" (BMV + USA + ETFs + cripto + FIBRAs, ~94 nombres).
    index_options: pares (nombre, símbolo yfinance) ofrecidos como "comprar el índice"."""
    symbols = _bmv_symbols() if universe == "bmv" else _all_symbols()
    frames = {name: _bars(sym) for name, sym in symbols.items()}
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 5}
    for idx_name, idx_symbol in index_options:
        idx_df = _bars(idx_symbol)
        if idx_df is not None and len(idx_df) > 5:
            frames[idx_name] = idx_df
    if asof is None:
        asof = max(df.index.max() for df in frames.values())
    asof = pd.Timestamp(asof).tz_localize(None)
    index_names = {n for n, _ in index_options}

    news = _news_by_ticker(str(asof.date()))
    fund_df = _load_fundamentals()
    divs_all = _load_dividends()
    eps_all = _load_reported_eps()
    macro = _macro_context(asof)
    regime = _regime(asof)
    exclude = exclude or set()

    # Primera pasada: junta métricas crudas de todas las emisoras para poder
    # calcular z-scores cruzados (comparables entre sí) en la segunda pasada.
    metrics: dict[str, dict] = {}
    for name, df in frames.items():
        if name in exclude:
            continue
        hist = df.loc[:asof]
        if hist.empty:
            continue
        close = float(hist["Close"].iloc[-1])
        day = _pct(close, float(hist["Close"].iloc[-2])) if len(hist) > 1 else "na"
        week = _pct(close, float(hist["Close"].iloc[-6])) if len(hist) > 5 else "na"
        month = _pct(close, float(hist["Close"].iloc[-21])) if len(hist) > 21 else "na"
        year, year_raw = "na", None
        if len(hist) > 252 and float(hist["Close"].iloc[-252]) > 0 and float(hist["Close"].iloc[-21]) > 0:
            p21, p252 = float(hist["Close"].iloc[-21]), float(hist["Close"].iloc[-252])
            year = _pct(p21, p252)
            year_raw = p21 / p252 - 1

        window_52w = hist["Close"].tail(252)
        hi52 = float(window_52w.max()) if len(window_52w) else close
        lo52 = float(window_52w.min()) if len(window_52w) else close
        from_hi = _pct(close, hi52)
        from_lo = _pct(close, lo52) if lo52 > 0 else "na"

        rets = hist["Close"].pct_change().tail(20)
        vol20 = float(rets.std() * (252 ** 0.5) * 100) if len(rets) > 5 else None

        stop = atr_stop(hist, close, mult=stop_mult)
        fund = _fundamentals_asof(fund_df, name, asof)
        div_y = _div_yield_ttm(name, close, asof, divs_all)
        symbol = symbols.get(name, name)
        eps_y = _earnings_yield_ttm(symbol, close, asof, eps_all)

        item = news.get(name) or {}
        metrics[name] = {
            "close": close, "day": day, "week": week, "month": month, "year": year,
            "year_raw": year_raw, "from_hi": from_hi, "from_lo": from_lo, "vol20": vol20,
            "stop": stop, "fund": fund, "div_y": div_y, "eps_y": eps_y,
            "n_art": item.get("n_articles", 0), "sent": item.get("news_sent_score"),
        }

    scores = _composite_scores(metrics)
    universo_txt = (
        "de cada emisora de la BMV" if universe == "bmv"
        else "del universo completo disponible (BMV, acciones USA, ETFs, cripto y FIBRAs)"
    )
    idx_lista = ", ".join(n for n, _ in index_options)

    lines = [
        "Eres el inversor más chingón de México: agresivo cuando hay evidencia, "
        f"frío y sin miedo a quedarte en cash cuando no la hay. Tienes 50 dólares de un libro "
        f"de paper trading y una vista de 360° {universo_txt} abajo, una fila por "
        "emisora. Usa TODO el dato: precio, momentum a varios plazos, fundamentales, "
        "dividendos, noticias y el régimen de mercado. No hay reglas externas que te frenen: "
        "la decisión es completamente tuya, justificada con los números que ves.",
        "",
        f"Cierre {asof.date()}. Régimen de mercado (HMM sobre IPC/S&P500): {regime}.",
        "Macro 5 días: " + ", ".join(f"{k} {v}" for k, v in macro.items()) if macro else "Macro: sin dato.",
        "Contexto de los modelos cuantitativos: ningún factor técnico propio (WoE de patrones, "
        "50 features extra, order-flow) pasó Information Value >= 0.02 en backtest (máximo visto "
        "0.0046); tómalos como ruido, no como señal fuerte. El LSTM de secuencia tiene AUC "
        "~0.50-0.54 (casi azar). El factor que SÍ mostró ganancia pre-costos en backtest es "
        "momentum 12-1 (retorno de 12 meses sin el último mes) — en universo amplio (con cripto) "
        "mostró ganancia incluso neta de costos; en BMV sola, casi todo se lo come el costo de "
        "transacción y perdió contra el índice en pruebas de 24 meses.",
        "Cada emisora trae un 'score' multifactor ya calculado (no lo calcules tú): combina "
        "momentum 12-1 (40%), calidad/margen operativo (25%), baja volatilidad realizada (20%) "
        "y valor/earnings yield (15%), todo normalizado en z-score contra el resto del universo "
        "ese día. Está ordenado de mayor a menor score: entre más arriba, mejor perfil combinado. "
        "Úsalo como punto de partida, no como regla ciega — tú decides con el resto de columnas "
        "y las noticias si de verdad conviene.",
        f"{idx_lista} está(n) en la lista como posición(es) de índice: comprarlo es "
        "diversificación total, cero riesgo idiosincrático de una sola emisora. En pruebas "
        "históricas de 24 meses, elegir pocas acciones individuales de la BMV le perdió al "
        "índice la mayoría de los periodos. Si no hay convicción clara y fuerte en nombres "
        "puntuales, mete la mayor parte o todo el libro al índice en vez de forzar un pick.",
        "Columnas por emisora: score, precio, retorno día, semana, mes, año (12-1 momentum), "
        "distancia a máximo/mínimo de 52 semanas, volatilidad realizada 20d anualizada, "
        "stop de referencia (2.5x ATR14, pensado para sostener semanas/un mes, no para "
        "day-trading), margen bruto/operativo/neto más reciente, crecimiento de ingresos y "
        "utilidad YoY, dividend yield TTM, earnings yield TTM (UPA TTM / precio), "
        "número de noticias del día y su sentimiento.",
    ]
    if held:
        tenencia = ", ".join(f"{t} ({d}d)" for t, d in held.items())
        lines.append(
            f"Posiciones abiertas hoy en el libro y días que llevan: {tenencia}. "
            "Rotar de nombre tiene un costo real de transacción (~0.55% ida y vuelta); "
            "si la posición actual sigue teniendo la misma evidencia que la justificó, "
            "mantenerla es gratis, cambiarla no."
        )
    if exclude:
        lines.append(
            "En enfriamiento (tocaron su stop hace poco, no disponibles este periodo para "
            "forzar diversificación): " + ", ".join(sorted(exclude)) + "."
        )
    lines.append("")

    def pct_fmt(v):
        return "na" if v is None else f"{v * 100:+.1f}%"

    for name in sorted(metrics, key=lambda t: -scores.get(t, 0.0)):
        m = metrics[name]
        fund = m["fund"]
        sent_txt = "na" if m["sent"] is None else f"{float(m['sent']):+.2f}"
        vol_txt = "na" if m["vol20"] is None else f"{m['vol20']:.0f}%"
        line = (
            f"{name} score {scores.get(name, 0.0):+.2f} {m['close']:.2f} "
            f"dia {m['day']} sem {m['week']} mes {m['month']} año {m['year']} "
            f"vs52whi {m['from_hi']} vs52wlo {m['from_lo']} "
            f"vol20 {vol_txt} "
            f"stop {m['stop']:.2f} "
            f"margB {pct_fmt(fund.get('gross_margin'))} margO {pct_fmt(fund.get('oper_margin'))} "
            f"margN {pct_fmt(fund.get('net_margin'))} ventasYoY {pct_fmt(fund.get('revenue_yoy'))} "
            f"utilYoY {pct_fmt(fund.get('earnings_yoy'))} divYield {pct_fmt(m['div_y'])} "
            f"earnYield {pct_fmt(m['eps_y'])} noticias {m['n_art']} sent {sent_txt}"
        )
        if name in index_names:
            line += " [ÍNDICE]"
        if held and name in held:
            line += f" [YA LA TIENES, {held[name]}d]"
        lines.append(line)

    lines.append("")
    lines.append(
        f"Reparte 50 dólares. Solo compras (long only), el efectivo es una posición válida y "
        f"{idx_lista} (el/los índice(s)) también cuenta(n) como posición. Si vas con nombres "
        "individuales, diversifica entre 4 y 7, cada uno entre 5 y 15 dólares — concentrar todo "
        "en 2-3 te expone a riesgo idiosincrático sin evidencia de que le gane al índice. Si "
        "prefieres el índice, puedes ponerle la mayor parte o todo el libro ahí, es una sola "
        "línea. Decide con los datos, no por default: si de verdad nada individual convence, "
        "prioriza el índice o cash antes que forzar un pick débil."
    )
    if cost_note:
        lines.append(
            "Cada vez que vendes un nombre que ya tenías o compras uno nuevo pagas ~0.55% "
            "de costo de transacción sobre ese monto. No rotes de nombre solo por un ligero "
            "cambio de ranking: solo cambia una posición existente si la evidencia nueva es "
            "claramente mejor que la que la justificó, o si tocó su stop."
        )
    lines.append(
        "Responde solo JSON {\"allocations\":[{\"ticker\",\"usd\",\"horizon\"}],\"cash_usd\",\"motivo\"}."
    )
    return "\n".join(lines)
