"""Agente de dominio: Features/ML — la bitácora completa de modelos
entrenados, no solo lo que está en producción.

Distinción que SIEMPRE debe mantener al responder:
  - En producción hoy: SuperModel vía scripts/record_signals.py
    (data/signals/history/, append-only, track record real).
  - Banco de experimentos (investigación, NO decide el libro):
    ranking LightGBM (scripts/generate_all_signals.py, se reentrena en
    cada corrida, sin persistencia), WoE, LSTM, TCN, régimen/HMM, Ising,
    sentimiento de noticias como predictor — ver data/eval/agent_scorecard.md
    y docs/SISTEMA_END_TO_END.md §9 para los veredictos reales.

TODO equipo Features/ML: implementar query_features_ml de verdad sobre
data/eval/trial_log.jsonl, agent_scorecard.md y data/signals/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORTFOLIO_COV_PATH = ROOT / "data" / "eval" / "forecast_demo" / "portfolio_optimo_cov.json"
AGENT_SCORECARD_PATH = ROOT / "data" / "eval" / "agent_scorecard.md"
TRIAL_LOG_PATH = ROOT / "data" / "eval" / "trial_log.jsonl"

SYSTEM_PROMPT = """Eres el agente de dominio de Features/ML. No expongas
"el modelo" como si fuera uno solo: expón el registro completo de
experimentos, incluyendo los que no aportaron (WoE, LSTM, Ising, etc.), con
su veredicto real medido. Distingue siempre entre lo que opera en
producción (SuperModel) y lo que es investigación abierta o descartada."""


@tool(
    "schema_features_ml",
    "Describe el inventario de modelos/experimentos y qué está en producción.",
    {},
)
async def schema_features_ml(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "En producción: SuperModel (src/ml/super_model.py), corre "
                "diario vía scripts/record_signals.py, historial append-only "
                "en data/signals/history/YYYY-MM-DD.json. "
                "Banco de experimentos (NO decide el libro — "
                "docs/SISTEMA_END_TO_END.md §9): ranking LightGBM "
                "(scripts/generate_all_signals.py, se reentrena cada corrida), "
                "WoE (IV máx 0.0046), LSTM/TCN (AUC 0.50-0.54), régimen HMM, "
                "Ising, sentimiento de noticias. Veredictos reales en "
                "data/eval/agent_scorecard.md y data/eval/trial_log.jsonl. "
                "Validación anti-leakage: src/ml/purged_cv.py "
                "(purge + embargo, López de Prado)."
            ),
        }]
    }


@tool(
    "query_features_ml",
    "Responde una pregunta en lenguaje natural sobre la bitácora de modelos/experimentos, "
    "leyendo data/eval/agent_scorecard.md y data/eval/trial_log.jsonl.",
    {"pregunta": str},
)
async def query_features_ml(args: dict) -> dict:
    pregunta = args.get("pregunta", "")
    partes: list[str] = []
    partes.append(
        f"Pregunta recibida: {pregunta!r}" if pregunta
        else "Bitácora completa de modelos/experimentos:"
    )

    if AGENT_SCORECARD_PATH.exists():
        partes.append(
            "--- data/eval/agent_scorecard.md ---\n"
            + AGENT_SCORECARD_PATH.read_text(encoding="utf-8").strip()
        )
    else:
        partes.append(f"No se encontró {AGENT_SCORECARD_PATH} — scorecard no generado todavía.")

    if TRIAL_LOG_PATH.exists():
        lineas = [l for l in TRIAL_LOG_PATH.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        resumen = [f"--- data/eval/trial_log.jsonl: {len(lineas)} corridas registradas ---"]
        for linea in lineas[-5:]:
            try:
                rec = json.loads(linea)
            except json.JSONDecodeError:
                continue
            resumen.append(
                f"- {rec.get('ts', '?')} family={rec.get('family', '?')} hash={rec.get('hash', '?')}"
            )
        partes.append("\n".join(resumen))
    else:
        partes.append(f"No se encontró {TRIAL_LOG_PATH}.")

    return {"content": [{"type": "text", "text": "\n\n".join(partes)}]}


@tool(
    "pronostico_precio",
    "Pronóstico de retorno esperado a un horizonte (días hábiles) para un ticker, "
    "corriendo LOS 4 INTENTOS de scripts/forecast_horizons.py en modo 'sin ancla' "
    "(lightgbm_ranking, momentum_12_1, woe_score, regimen_hmm — régimen de Markov del "
    "IPC/S&P500), no solo uno. Soporta cualquier ticker con datos en data/market/ "
    "(~95 tickers BMV+USA+ETF+cripto+FIBRA, ver src/data/store.py ALL_SYMBOLS) "
    "siempre que tenga suficiente historia (300+ sesiones). El horizonte se ajusta "
    "al más cercano disponible entre 30/60/90d (21/42/63 sesiones).",
    {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker a pronosticar, ej. WALMEX o SPY."},
            "horizonte_dias": {
                "type": "integer",
                "description": "Horizonte deseado en sesiones (días hábiles). Opcional, default 42 (~60d).",
            },
        },
        "required": ["ticker"],
    },
)
async def pronostico_precio(args: dict) -> dict:
    ticker = str(args.get("ticker") or "").strip().upper()
    horizonte_dias = args.get("horizonte_dias") or 42

    from scripts.forecast_horizons import HORIZONS, TICKERS, run_sin_ancla

    if ticker not in TICKERS:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"{ticker} no está en el catálogo del store "
                    f"(src/data/store.py ALL_SYMBOLS) — no hay datos descargados "
                    f"para ese símbolo todavía. Usa el nombre del store (ej. "
                    f"WALMEX, no WALMEXV.MX)."
                ),
            }]
        }

    horizon_label, horizon_days = min(HORIZONS.items(), key=lambda kv: abs(kv[1] - int(horizonte_dias)))

    MODELOS = ("lightgbm_ranking", "momentum_12_1", "woe_score", "regimen_hmm")
    HONESTIDAD = {
        "lightgbm_ranking": "a 1 día no tiene skill probado (hit rate ~50%); la única "
            "evidencia real a favor es SPY a 60-90d.",
        "momentum_12_1": "señal de momentum clásica (ret_20d); sin ventaja probada "
            "consistente fuera de SPY a mediano plazo.",
        "woe_score": "Information Value máximo medido fue muy bajo (~0.0046) — "
            "trátalo como señal débil en general.",
        "regimen_hmm": "detecta régimen bull/lateral/bear del IPC/S&P500 (cadena de "
            "Markov), el retorno condicional por estado es aproximado, no exacto.",
    }

    resultados = []
    errores = []
    for modelo in MODELOS:
        try:
            r = run_sin_ancla(ticker, modelo, horizon_label, horizon_days)
        except Exception as exc:
            errores.append(f"- {modelo}: error ({exc})")
            continue
        if r is None:
            errores.append(f"- {modelo}: datos insuficientes para ajustar")
            continue
        price0 = r["price0"]
        price_final = float(r["daily"]["price_model"].iloc[-1])
        retorno_pct = (price_final / price0 - 1) * 100
        mejora_vs_naive_pct = (
            (r["mae_naive"] - r["mae_model"]) / r["mae_naive"] * 100 if r["mae_naive"] else 0.0
        )
        resultados.append(
            f"- **{modelo}**: retorno esperado {retorno_pct:+.2f}% (corte {r['cutoff_date'].date()}), "
            f"MAE {r['mae_model']:.2f} vs buy&hold {r['mae_naive']:.2f} "
            f"({mejora_vs_naive_pct:+.1f}% de error vs. no hacer nada). "
            f"Honestidad: {HONESTIDAD[modelo]}"
        )

    if not resultados:
        return {
            "content": [{
                "type": "text",
                "text": f"Ningún intento pudo correr para {ticker}:\n" + "\n".join(errores),
            }]
        }

    retornos = [float(l.split("retorno esperado ")[1].split("%")[0]) for l in resultados]
    promedio = sum(retornos) / len(retornos)
    dispersion = max(retornos) - min(retornos)

    texto = (
        f"Pronóstico {ticker} a {horizon_label} (~{horizon_days} sesiones), "
        f"4 intentos en modo 'sin ancla' (cada uno compone SOLO su propia predicción, "
        f"sin copiar el precio real día a día):\n\n" + "\n".join(resultados)
    )
    if errores:
        texto += "\n\nNo corrieron:\n" + "\n".join(errores)
    texto += (
        f"\n\nPromedio de los {len(resultados)} intentos: {promedio:+.2f}%. "
        f"Dispersión entre el más optimista y el más pesimista: {dispersion:.2f} pp — "
        + ("ALTA, los modelos NO coinciden, trátalo como incertidumbre real, no la "
           "promedies ciegamente." if dispersion > 3 else "baja, los modelos coinciden "
           "razonablemente entre sí.")
    )
    return {"content": [{"type": "text", "text": texto}]}


@tool(
    "recomendar_portafolio",
    "Resume la recomendación de portafolio óptimo (10 tickers, covarianza real, "
    "lambda=0.01) ya calculada en data/eval/forecast_demo/portfolio_optimo_cov.json.",
    {},
)
async def recomendar_portafolio(args: dict) -> dict:
    if not PORTFOLIO_COV_PATH.exists():
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"No existe {PORTFOLIO_COV_PATH}. Hay que correr "
                    "scripts/portfolio_optimizer_cov.py primero para generarlo."
                ),
            }]
        }

    with open(PORTFOLIO_COV_PATH, encoding="utf-8") as f:
        data = json.load(f)

    pesos = data.get("pesos", {})
    pesos_ordenados = sorted(pesos.items(), key=lambda kv: -kv[1])
    lineas = [f"- {t}: {w * 100:.1f}%" for t, w in pesos_ordenados if w > 0.001]
    sin_asignar = [t for t, w in pesos_ordenados if w <= 0.001]

    texto = (
        f"Portafolio recomendado (covarianza real, lambda={data.get('lambda')}, "
        f"10 tickers BMV+US, tope 25% por ticker):\n" + "\n".join(lineas)
    )
    if sin_asignar:
        texto += f"\n- (sin asignación, 0%): {', '.join(sin_asignar)}"
    texto += (
        f"\n\nRetorno esperado del portafolio a 21 días: "
        f"{data.get('retorno_portafolio_pct'):+.2f}%\n"
        f"Riesgo del portafolio a 21 días (desviación estándar, con covarianza cruzada "
        f"real entre tickers): {data.get('riesgo_portafolio_pct'):.2f}%\n\n"
        "Nota de honestidad: esta recomendación viene de modelos con evidencia limitada "
        "(ver query_features_ml / agent_scorecard.md — el ranking LightGBM no supera "
        "claramente al universo a 20 días, hit rate ~46%). Es una optimización matemática "
        "sobre retornos esperados imperfectos, no una garantía de rendimiento."
    )
    return {"content": [{"type": "text", "text": texto}]}


features_ml_agent = DomainAgent(
    name="features-ml",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_features_ml, query_features_ml, pronostico_precio, recomendar_portafolio],
)
