# Sistema end-to-end: libro de 50 USD (núcleo + satélite)

> Cómo funciona el sistema hoy (10-oct-2026), de los datos a la calificación del
> resultado. Desde esta fecha **los pesos los pone una regla fija**
> (`src/agent/core_satellite.py`) y Claude Haiku solo puede vetar emisoras por
> noticias. El diseño anterior, con Haiku como asignador, queda en §10 como
> historia.

---

## 1. Qué hace el sistema, en una frase

Cada fin de mes reparte un libro de paper trading de **50 USD**: 75% en el mercado
completo (SPY y NAFTRAC), 25% en 2 emisoras mexicanas elegidas por fórmula. Un
filtro de tendencia protege contra caídas grandes y Haiku hace de portero ante
malas noticias. Al mes siguiente se califica cómo le fue.

```
 DATOS                 REGLA (fin de mes)              PORTERO        LIBRO / LAB
 ─────                 ──────────────────              ───────        ───────────
 precios (yfinance) ─┐
 MXN=X ──────────────┤   núcleo: SPY 50% + NAFTRAC 25%
 fundamentales PIT ──┼─► satélite: ranking factorial ──► veto Haiku ──► plan de entrada
 UPA / dividendos ───┤   filtro de tendencia 10 meses   (solo quita)    lab.decision
 titulares ──────────┘   mínimo por orden, BMV entera                   lab.entry_plan
                                                                           │ ~20 días hábiles
                                                                           ▼
                                                                       lab.outcome (score)
```

---

## 2. Qué se puede comprar

Los universos viven en `src/data/universes.py` (BMV, USA, ETFs, cripto, FIBRAs: 94
tickers). La regla usa solo una parte:

| Bloque | Instrumentos | Por qué |
|---|---|---|
| Núcleo | SPY, NAFTRAC | Capturan el mercado. La brecha contra SPY era exposición, no puntería. |
| Satélite | ~36 emisoras BMV + 5 FIBRAs | Donde hay fundamentales point-in-time. |
| Fuera | Acciones USA sueltas, ETFs sectoriales, cripto | Las acciones USA ya vienen dentro de SPY. Cripto es una sola apuesta a BTC, no un factor. |

---

## 3. Datos

| Dato | Dónde | Cómo se usa |
|---|---|---|
| Precios OHLC diarios | `data/market/{SYMBOL}.parquet` (local); en Cloud Run se bajan de yfinance | Tendencia, volatilidad, momentum, precio de entrada |
| Tipo de cambio | `MXN=X` | Todo el libro se mide en USD; lo que cotiza en pesos se convierte |
| Fundamentales | `data/fundamentals.parquet`, con `available_date` | Margen bruto (rentabilidad), solo lo publicado hasta la fecha de decisión |
| UPA reportada | `data/eps/reported_eps.json` | Earnings yield TTM (valor) |
| Dividendos | `data/dividends/dividends.json` | Dividend yield TTM, cuando no hay UPA (FIBRAs) |
| Titulares | `data/news/history/articles.parquet` + snapshots de DataBursátil | Entrada del veto de Haiku |

La ingesta diaria (§6) sigue corriendo todos los días hábiles. Recolecta y
versiona datos, pero ya no dispara operaciones.

---

## 4. La regla (`src/agent/core_satellite.py`)

Los parámetros se fijaron **una sola vez**, por razones económicas y antes de ver
resultados. Cambiar uno es una prueba nueva (ver §8).

### 4.1 Núcleo (75%)
- SPY 50% y NAFTRAC 25%.
- Si un ETF cierra el mes **debajo de su media de 10 meses** (210 sesiones, con
  banda de ±1%), se queda a la mitad y el resto pasa a efectivo. Es un seguro
  contra caídas, no una fuente de rendimiento.

### 4.2 Satélite (25%)
1. **Score** de cada emisora: un rango percentil por factor, sin z-scores, para
   que un dato extremo no domine.

   | Factor | Peso | Medida |
   |---|---|---|
   | Rentabilidad | 30% | Margen bruto point-in-time |
   | Valor | 30% | Earnings yield TTM (o dividend yield si no hay UPA) |
   | Baja volatilidad | 25% | −volatilidad de 252 días |
   | Momentum | 15% | Retorno 12-1 dividido entre la volatilidad |

   El momentum bajó de 40% a 15% porque es el factor con peor evidencia en
   emisoras grandes de México y LatAm.
2. **Cuántos nombres:** los que quepan sin bajar del mínimo de ~100 MXN por orden.
   Con 50 USD son 2 (unos 6.25 USD cada uno).
3. **Histéresis:** una emisora entra si está en el top 20%, pero se mantiene
   hasta que cae debajo del top 40%. Así no se rota por ruido.
4. **Reselección trimestral** (ene, abr, jul, oct). Los otros meses solo se
   revisan la tendencia y la deriva; si sale un nombre, su lugar queda en
   efectivo hasta el trimestre.
5. **Tendencia:** una emisora debajo de su media de 10 meses no entra, o sale.
6. **BMV en acciones enteras:** si una sola acción no cabe en el slot, se brinca.
7. **Equiponderado:** cada nombre lleva la misma parte del 25%.

### 4.3 Riesgo
- **Stop de catástrofe** solo en el satélite: si el cierre queda 15% debajo del
  máximo desde la entrada, se vende en la apertura siguiente. Ya no existe el
  stop intramensual de 1.5×ATR, que provocaba whipsaw.
- **Banda de deriva:** una posición se rebalancea solo si su peso se movió más de
  5 pp, y solo hasta el borde de la banda.

### 4.4 La tendencia no guarda estado
El estado de la tendencia (dentro o fuera) se reconstruye con los últimos 12
cierres de mes de cada precio. Por eso el backtest y el job de Cloud Run, que no
guarda nada entre corridas, toman exactamente la misma decisión.

---

## 5. Claude Haiku como portero (`src/agent/veto.py`)

Haiku **no reparte dinero**. Recibe los candidatos del satélite y sus titulares
de los últimos 30 días (solo los publicados hasta la fecha de decisión) y, para
cada uno, devuelve:

- `evento`: recorte de distribución, cambio de guía, M&A, regulatorio,
  legal/fraude, resultados, deuda/calificación, otro, ninguno.
- `direccion`: negativa, neutral o positiva.
- `veto`: verdadero solo si el evento es material y negativo.

Protecciones contra los sesgos documentados de los LLM:
- **Anonimización:** las emisoras se llaman E1, E2…, y el ticker y los alias de la
  empresa se borran del texto.
- **Orden barajado** en cada réplica: 5 réplicas por fecha, con veto por mayoría.
- **Sin titulares no hay veto:** sin noticias, el nombre ni se manda.
- **Auditoría:** cada llamada queda en `data/eval/veto_log.jsonl` con el modelo,
  el hash del prompt y el de la respuesta.

Si Haiku veta un nombre, entra el siguiente del ranking. Si no hay
`ANTHROPIC_API_KEY`, la regla corre sola.

**El gemelo:** cada decisión guarda también lo que la regla habría hecho sin
Haiku (`raw.gemelo`). El valor de Haiku se mide como *libro con veto − gemelo*,
solo en paper trading hacia adelante, porque casi no hay titulares históricos
(solo WALMEX, 2018-2024).

---

## 6. Producción: Cloud Run + Postgres (`scripts/gcp_daily_job.py`)

Tres Cloud Run Jobs en `domesticospro`, hora `America/Mexico_City`, sobre Cloud SQL
Postgres (`mercado`, esquema `lab`):

| Horario | Job | Qué hace hoy |
|---|---|---|
| lun–vie 18:30 (sáb/dom, solo noticias) | `ingest` | Escribe `lab.price_daily` y `lab.news_daily` |
| lun–vie 19:15 | `decide` | **Solo actúa en la última sesión del mes.** Los demás días registra "no es cierre de mes" y sale. El día de rebalanceo baja ~15 meses de historia, corre la regla, el veto (si hay key) y el gemelo, y escribe `lab.decision` + `lab.entry_plan` para la apertura del siguiente día hábil. `--force` lo obliga a decidir cualquier día. |
| lun–vie 20:00 | `score` | Califica cada plan cuando se cumplen sus 20 días hábiles y escribe `lab.outcome` |

Qué guarda cada decisión:
- `lab.entry_plan`: una fila por posición (ticker, USD, horizonte `20d`, stop de
  catástrofe en USD para el satélite y vacío para el núcleo, precio de entrada).
- `lab.decision.raw`: los parámetros de la regla (`regla`), los pesos sin veto
  (`gemelo`), los vetos con su motivo, los candidatos y el estado de tendencia.

La imagen (`deploy/cloudrun/Dockerfile`) incluye `data/fundamentals.parquet`,
`data/eps/` y `data/dividends/`, para que el satélite use los mismos factores que
en el backtest. Sin Postgres, los planes caen a `data/lab/entry_plans.jsonl`.

Matiz conocido: el `score` revisa el stop contra el **mínimo** del día, mientras
que la regla lo evalúa al **cierre**. La calificación queda un poco más estricta
que la regla.

---

## 7. Backtest (`scripts/backtest_core_satellite.py`)

El mismo motor sirve para tres políticas:

| `--policy` | Qué es |
|---|---|
| `mecanico` | La regla sola: el gemelo |
| `veto` | La regla con el veto de Haiku |
| `llm` | Haiku repartiendo los 50 como antes, para comparar |

Ejecución honesta:
- Señal al cierre de la última sesión del mes; ejecución en la **apertura** del
  día siguiente.
- Precios MX convertidos a USD, así que la divisa entra al resultado.
- BMV en acciones enteras y mínimo de ~100 MXN por orden.
- Costo de **0.6% ida y vuelta** (0.8% con `--stress`), más 0.4% de conversión en
  lo que se compra en pesos.
- Benchmarks: SPY, NAFTRAC en USD y la **mezcla** w·NAFTRAC + (1−w)·SPY, donde w
  es el peso promedio en México del libro. Todo se reporta en USD y en MXN.
- **Atribución** del P&L en núcleo y satélite, local y divisa, más costos; las
  partes suman exacto al equity final.

### Resultado actual (24 meses, oct-2024 → oct-2026)

| | Libro | SPY | Mezcla (31% NAFTRAC) |
|---|---|---|---|
| Rendimiento USD | **+21.5%** | +34.9% | +35.4% |
| Rendimiento MXN | +14.2% | +26.8% | +27.3% |
| Sharpe (USD) | 0.97 | 0.99 | 1.13 |
| Caída máxima (USD) | **−12.4%** | −19.0% | −15.9% |

- Meses con ganancia: 17 de 25. Meses en que le ganó a SPY: 11 de 25. A la
  mezcla: 10 de 25.
- Atribución en USD: núcleo +8.33, satélite +3.00 (de los cuales +1.78 por
  divisa), costos −0.57.
- El rezago viene del efectivo ocioso (88% invertido en promedio) y del filtro de
  tendencia: recortó el núcleo en 11 de 25 meses, justo antes de los rebotes de
  abril 2025 y abril 2026. A cambio, el libro cae menos.
- Con 17 pruebas sobre esta historia, el azar solo ya da un Sharpe de ~1.30. El
  Deflated Sharpe sale en **0.32**, lejos del 0.95 que pediría evidencia. **No hay
  habilidad demostrada**, y con 24 meses tampoco podría haberla.

---

## 8. Disciplina estadística (`src/ml/trial_stats.py`)

- **Bitácora de pruebas** (`data/eval/trial_log.jsonl`): cada corrida del backtest
  se anota con el hash de su configuración. Configuraciones distintas suben N;
  cambiar solo los supuestos de costo no lo sube. `--prior-trials 15` suma las
  variantes con LLM que se probaron antes de la bitácora.
- **Deflated Sharpe:** probabilidad de que el Sharpe real supere el máximo
  esperado por azar entre N pruebas, corregida por asimetría y curtosis.
  > 0.95 sería evidencia.
- **Regla de trabajo:** no se ajustan parámetros mirando el backtest. La
  evaluación que cuenta es el paper trading hacia adelante, libro contra gemelo.
  Ojo: el `deflated_sharpe_ratio` viejo de `src/ml/purged_cv.py` tiene la escala
  mal; usa el de `trial_stats.py`.

---

## 9. Lo que sigue en el repo pero ya no decide

| Pieza | Estado |
|---|---|
| `scripts/generate_all_signals.py` → `data/signals/*.json` | Sigue generando rankings LightGBM; el `decide` ya no los usa para asignar |
| WoE, LSTM, régimen HMM, Ising (`src/ml/`) | Investigación. Veredicto medido: nada técnico propio separa (IV máximo 0.0046; AUC del LSTM 0.50-0.54) |
| `src/agent/investor.py`, `tools.py`, `policy.py`, `book.py` | El agente con tools de 8 turnos. Ya no lo llama el job |
| `src/agent/bmv_report.py` | El reporte 360°. Lo usan la política `llm` del backtest y los loaders de factores |
| `_plan_from_agent`, `_plan_from_llm`, `_plan_from_rules` en el job | Código muerto; se puede borrar |
| `scripts/backtest_bmv_walkforward.py`, `backtest_bmv_month.py` | Backtests viejos: ignoran la divisa y cobran ~1.1% ida y vuelta. Reemplazados por `backtest_core_satellite.py` |
| `src/ml/risk_engine.py` (Kelly, ATR) | Ya no se usa para el libro |
| ETL legado de SQL Server (`src/etl/`) | Arquitectura original, reemplazada por parquet + Postgres `lab` |

---

## 10. Historia: por qué se sacó a Haiku del volante

Antes, Haiku recibía un reporte 360° y repartía los 50 USD con libertad
(score 40% momentum, stop de 1.5-2.5×ATR). Resultados del walk-forward viejo:

| Configuración | Resultado | Benchmark |
|---|---|---|
| 12m, stop 1.5×ATR | −14.7% | NAFTRAC +7.7% |
| 12m, mismo plan sin stop | +14.4% | — |
| 24m, solo BMV, 2-3 nombres | −22.0% | NAFTRAC +23.2% |
| 24m, BMV + NAFTRAC exento de stop | −8.4% | NAFTRAC +23.2% |
| 24m, universo completo + SPY/NAFTRAC | +5.7% | SPY +35.9% |

Además, ese backtest no convertía la divisa y cobraba el doble de costos.

La revisión de literatura 2023-2026 (FINSABER, StockBench, Bailey et al.,
Lopez-Lira, Kaminski y Lo) llevó a cinco cambios:
1. Núcleo pasivo con un satélite chico.
2. El LLM extrae y veta, no asigna.
3. Menos momentum.
4. Filtro de tendencia mensual en vez del stop ATR.
5. Dejar de iterar sobre los mismos 24 meses y preregistrar.

---

## 11. Resumen por capa

| Capa | Qué hace | Archivo ancla |
|---|---|---|
| Qué se puede comprar | Núcleo SPY/NAFTRAC + satélite BMV/FIBRAs | `src/agent/core_satellite.py`, `src/data/universes.py` |
| Con qué datos | Precios, FX, fundamentales PIT, UPA, dividendos, titulares | `data/`, `scripts/gcp_daily_job.py` (`_cs_frames`) |
| Quién pone los pesos | La regla fija | `src/agent/core_satellite.py` |
| Qué hace el LLM | Clasifica noticias y veta; nunca pone pesos | `src/agent/veto.py` |
| Cuándo decide | Última sesión del mes; ejecución en la apertura siguiente | `scripts/gcp_daily_job.py` (`run_decide`) |
| Dónde queda | Postgres `lab` (decisión, plan, gemelo, vetos) | `src/db/lab_store.py` |
| Cómo se califica | 20 días hábiles después, contra el precio real | `scripts/gcp_daily_job.py` (`run_score`) |
| Cómo se evalúa | Backtest honesto + bitácora + Deflated Sharpe; la evidencia real es libro contra gemelo hacia adelante | `scripts/backtest_core_satellite.py`, `src/ml/trial_stats.py` |
