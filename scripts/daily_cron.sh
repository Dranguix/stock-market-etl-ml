#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Cron diario BMV — Track Record Builder
#  Corre cada día hábil antes de la apertura (07:30 CST = 13:30 UTC)
#
#  Instalar:
#    chmod +x scripts/daily_cron.sh
#    crontab -e
#    # Añadir esta línea (07:30 hora Ciudad de México, lunes-viernes):
#    30 7 * * 1-5 /Users/rod/Projects/stock-market/scripts/daily_cron.sh >> /Users/rod/Projects/stock-market/logs/cron.log 2>&1
#
#  Verificar:
#    crontab -l
#    tail -f logs/cron.log
# ══════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/Users/rod/Projects/stock-market"
PYTHON="$REPO/venv/bin/python"
LOG_DIR="$REPO/logs"
DATE=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"

echo ""
echo "════════════════════════════════════════"
echo "  Cron BMV — $DATE $(date +%H:%M:%S)"
echo "════════════════════════════════════════"

cd "$REPO"
source "$REPO/.env" 2>/dev/null || true

# ── 1. Snapshot de noticias + cables ────────────────────────────────────────
echo ""
echo "  [1/4] Snapshot noticias DataBursátil…"
$PYTHON scripts/databursatil_news_snapshot.py --rebuild-panel --universe bmv \
    && echo "  OK noticias" \
    || echo "  WARN: fallo snapshot noticias (continuando)"

# ── 2. Descarga de hechos del día anterior ──────────────────────────────────
# Calculamos el día hábil anterior (ayer o el viernes si hoy es lunes)
AYER=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d)
echo ""
echo "  [2/4] Hechos del $AYER…"
$PYTHON scripts/backfill_hechos.py --fecha "$AYER" \
    && echo "  OK hechos $AYER" \
    || echo "  WARN: fallo hechos $AYER (continuando)"

# ── 3. Actualizar precios OHLCV ─────────────────────────────────────────────
echo ""
echo "  [3/4] Actualizar precios BMV (yfinance)…"
$PYTHON -c "
from src.data import store
from src.data.universes import UNIVERSES
bmv = UNIVERSES.get('🇲🇽 Acciones BMV', {}).get('tickers', {})
macro = {'MXN=X': 'USDMXN', 'NAFTRAC.MX': 'NAFTRAC', 'CL=F': 'WTI', '^GSPC': 'SP500'}
for sym in list(bmv.values()) + list(macro.keys()):
    try:
        store.update(sym, verbose=False)
    except Exception as e:
        print(f'  skip {sym}: {e}')
print('  OK precios')
" || echo "  WARN: fallo actualización precios"

# ── 4. Registro de señales ────────────────────────────────────────────────
echo ""
echo "  [4/4] SuperModel + registro de señales…"
$PYTHON scripts/record_signals.py --no-gemini \
    && echo "  OK señales registradas" \
    || echo "  ERROR: fallo el modelo"

echo ""
echo "  Cron terminado $(date +%H:%M:%S)"
echo "════════════════════════════════════════"
