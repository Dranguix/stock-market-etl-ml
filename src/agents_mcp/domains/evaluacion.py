"""Agente de dominio: Evaluación/Riesgo — backtests honestos, Deflated
Sharpe y bitácora anti-overfitting.

Esquema real:
  - data/eval/trial_log.jsonl       bitácora de pruebas (hash de config, N)
  - src/ml/trial_stats.py           Deflated Sharpe correcto (el de
                                     src/ml/purged_cv.py tiene la escala mal)
  - scripts/backtest_core_satellite.py  el backtest honesto vigente
                                     (--policy mecanico|veto|llm)

Regla de trabajo que este agente debe respetar al responder: no se ajustan
parámetros mirando el backtest. La evidencia que cuenta es el paper trading
hacia adelante (libro contra gemelo), no el backtest histórico — dilo
explícitamente si alguien pregunta "¿cuál es la mejor configuración?".

TODO equipo Evaluación/Riesgo: implementar query_evaluacion de verdad sobre
trial_log.jsonl y los data/eval/*.json existentes.
"""
from __future__ import annotations

import json
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent

ROOT = Path(__file__).resolve().parents[3]
TRIAL_LOG = ROOT / "data" / "eval" / "trial_log.jsonl"
FORECAST_DEMO_DIR = ROOT / "data" / "eval" / "forecast_demo"
LLM_SIM_FILES = {
    "1m": FORECAST_DEMO_DIR / "llm_sim_1m.json",
    "2m": FORECAST_DEMO_DIR / "llm_sim_2m.json",
    "3m": FORECAST_DEMO_DIR / "llm_sim_3m.json",
}

DISCIPLINA = (
    "Nunca se ajustan parámetros mirando el backtest histórico: la evidencia "
    "que cuenta es el paper trading hacia adelante (libro contra gemelo), no "
    "el backtest."
)

SYSTEM_PROMPT = """Eres el agente de dominio de Evaluación/Riesgo:
backtests, Deflated Sharpe, bitácora de pruebas. Nunca recomiendes ajustar
un parámetro basado solo en el backtest — esa disciplina está documentada
y es intencional (ver trial_stats.py). Si preguntan por una métrica,
siempre aclara sobre cuántas pruebas (N) se calculó el Deflated Sharpe."""


@tool(
    "schema_evaluacion",
    "Describe la bitácora de pruebas y el motor de backtest del dominio Evaluación.",
    {},
)
async def schema_evaluacion(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "data/eval/trial_log.jsonl — una fila por corrida de "
                "backtest, con hash de su configuración (sube N solo con "
                "configuraciones distintas, no con cambios de costo). "
                "Deflated Sharpe correcto: src/ml/trial_stats.py (el de "
                "purged_cv.py tiene la escala mal, no usarlo). "
                "Backtest vigente: scripts/backtest_core_satellite.py "
                "(--policy mecanico=regla sola, veto=regla+Haiku, "
                "llm=Haiku repartiendo libre, histórico). "
                "Resultado actual (24m): libro +21.5% USD, Sharpe 0.97, "
                "Deflated Sharpe 0.32 — sin skill demostrado."
            ),
        }]
    }


def _count_trials() -> tuple[int, dict[str, int]]:
    """Cuenta filas totales en trial_log.jsonl y por familia."""
    if not TRIAL_LOG.exists():
        return 0, {}
    total = 0
    por_familia: dict[str, int] = {}
    for line in TRIAL_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fam = row.get("family", "desconocida")
        por_familia[fam] = por_familia.get(fam, 0) + 1
    return total, por_familia


def _resumen_sim(etiqueta: str, path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    libro = data.get("usd", {}).get("libro", {})
    benchmarks = {k: v for k, v in data.get("usd", {}).items() if k != "libro"}
    bench_txt = "; ".join(
        f"{nombre}: {info.get('retorno_pct')}%" for nombre, info in benchmarks.items()
    )
    dsr = data.get("dsr", {})
    n_trials = data.get("n_trials")
    return (
        f"{etiqueta} ({data.get('start')} a {data.get('end')}): "
        f"retorno del libro {libro.get('retorno_pct')}% (Sharpe {libro.get('sharpe')}), "
        f"benchmarks [{bench_txt}], "
        f"Deflated Sharpe {dsr.get('dsr')} (Sharpe anual {dsr.get('sharpe_anual')} "
        f"vs. azar esperado {dsr.get('sr0_anual')} sobre N={n_trials} pruebas, "
        f"{dsr.get('n_obs')} observaciones diarias)."
    )


@tool(
    "query_evaluacion",
    "Responde una pregunta en lenguaje natural sobre backtests/evaluación: "
    "cuántas pruebas hay en la bitácora y cómo le fue al agente LLM real en "
    "sus simulaciones de 30/60/90 días.",
    {"pregunta_tipo": str},
)
async def query_evaluacion(args: dict) -> dict:
    pregunta_tipo = (args.get("pregunta_tipo") or "resumen").strip().lower()

    total, por_familia = _count_trials()
    if total:
        detalle_familias = ", ".join(f"{fam}: {n}" for fam, n in sorted(por_familia.items()))
        trial_txt = (
            f"data/eval/trial_log.jsonl tiene {total} filas registradas "
            f"({detalle_familias})."
        )
    else:
        trial_txt = "data/eval/trial_log.jsonl no existe o está vacío todavía."

    partes = [trial_txt]

    if pregunta_tipo in ("resumen", "", "ultimo_backtest", "sims", "llm_sim"):
        resumenes = []
        for etiqueta, path in LLM_SIM_FILES.items():
            resumen = _resumen_sim(f"Simulación LLM {etiqueta}", path)
            if resumen:
                resumenes.append(resumen)
        if resumenes:
            partes.append(
                "Resultados reales del agente LLM de producción (forecast_demo):\n- "
                + "\n- ".join(resumenes)
            )
        else:
            partes.append(
                "No encontré los archivos llm_sim_1m/2m/3m.json en "
                "data/eval/forecast_demo/."
            )

    partes.append(DISCIPLINA)
    return {"content": [{"type": "text", "text": "\n\n".join(partes)}]}


evaluacion_agent = DomainAgent(
    name="evaluacion",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_evaluacion, query_evaluacion],
)
