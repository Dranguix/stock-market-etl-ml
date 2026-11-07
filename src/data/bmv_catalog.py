"""
Catálogo canónico de emisoras BMV líquidas — fuente única para OHLCV, intradía,
fundamentales y SQL ETL.

Claves internas (ej. WALMEX) alineadas con SuperModel y data/intraday/.
"""
from __future__ import annotations

# ticker interno → símbolo yfinance
BMV_YFINANCE: dict[str, str] = {
    "WALMEX": "WALMEX.MX",   "FEMSAUBD": "FEMSAUBD.MX", "GMEXICOB": "GMEXICOB.MX",
    "CEMEXCPO": "CEMEXCPO.MX", "GFNORTEO": "GFNORTEO.MX", "BIMBOA": "BIMBOA.MX",
    "ALSEA": "ALSEA.MX",     "KIMBERA": "KIMBERA.MX",   "GRUMAB": "GRUMAB.MX",
    "ASURB": "ASURB.MX",     "GAPB": "GAPB.MX",         "ORBIA": "ORBIA.MX",
    "AC": "AC.MX",           "VESTA": "VESTA.MX",       "PINFRA": "PINFRA.MX",
    "OMAB": "OMAB.MX",       "MEGACPO": "MEGACPO.MX",   "TLEVISA": "TLEVISACPO.MX",
    "BOLSAA": "BOLSAA.MX",   "GCARSOA1": "GCARSOA1.MX", "LABB": "LABB.MX",
    "CUERVO": "CUERVO.MX",   "HERDEZ": "HERDEZ.MX",     "GENTERA": "GENTERA.MX",
    "VOLARA": "VOLARA.MX",   "Q": "Q.MX",               "FUNO11": "FUNO11.MX",
    "GCC": "GCC.MX",         "AMXL": "AMXB.MX",         "CHDRAUIB": "CHDRAUIB.MX",
    "GFINBURO": "GFINBURO.MX", "VISTA": "VISTAA.MX",   "ALFAA": "ALFAA.MX",  # deslistada: no operable en GBM
    "KOFUBL": "KOFUBL.MX",   "HOTEL": "HOTEL.MX",       "POSADAS": "POSADASA.MX",
}

# ticker interno → emisora_serie DataBursátil / stage.cat_tickers
BMV_EMISORA_SERIE: dict[str, str] = {
    "WALMEX": "WALMEX*",   "FEMSAUBD": "FEMSAUBD", "GMEXICOB": "GMEXICOB",
    "CEMEXCPO": "CEMEXCPO", "GFNORTEO": "GFNORTEO", "BIMBOA": "BIMBOA",
    "ALSEA": "ALSEA*",     "KIMBERA": "KIMBERA",   "GRUMAB": "GRUMAB",
    "ASURB": "ASURB",      "GAPB": "GAPB",         "ORBIA": "ORBIA*",
    "AC": "AC*",           "VESTA": "VESTA*",      "PINFRA": "PINFRA*",
    "OMAB": "OMAB",        "MEGACPO": "MEGACPO",   "TLEVISA": "TLEVISACPO",
    "BOLSAA": "BOLSAA",    "GCARSOA1": "GCARSOA1", "LABB": "LABB",
    "CUERVO": "CUERVO*",   "HERDEZ": "HERDEZ*",     "GENTERA": "GENTERA*",
    "VOLARA": "VOLARA",    "Q": "Q*",               "FUNO11": "FUNO11",
    "GCC": "GCC*",         "AMXL": "AMXB",         "CHDRAUIB": "CHDRAUIB",
    "GFINBURO": "GFINBURO", "VISTA": "VISTAA",      "ALFAA": "ALFAA",
    "KOFUBL": "KOFUBL",    "HOTEL": "HOTEL*",       "POSADAS": "POSADASA",
}

# ticker interno → emisora /v2/financieros (sin serie)
BMV_EMISORA_FUNDAMENTALS: dict[str, str] = {
    "WALMEX": "WALMEX",   "FEMSAUBD": "FEMSA",   "GMEXICOB": "GMEXICO",
    "CEMEXCPO": "CEMEX",  "GFNORTEO": "GFNORTE", "BIMBOA": "BIMBO",
    "ALSEA": "ALSEA",     "KIMBERA": "KIMBER",   "PINFRA": "PINFRA",
    "ORBIA": "ORBIA",     "GRUMAB": "GRUMA",     "ASURB": "ASUR",
    "OMAB": "OMA",        "GAPB": "GAP",         "GCARSOA1": "GCARSO",
    "MEGACPO": "MEGA",    "LABB": "LAB",         "Q": "Q",
    "VOLARA": "VOLAR",    "CUERVO": "CUERVO",    "AC": "AC",
    "VESTA": "VESTA",     "VISTA": "VISTA",      "BOLSAA": "BOLSA",
    "GCC": "GCC",         "HOTEL": "HOTEL",      "POSADAS": "POSADAS",
    "TLEVISA": "TLEVISA", "HERDEZ": "HERDEZ",    "GENTERA": "GENTERA",
    "FUNO11": "FUNO",     "ALFAA": "ALFA",       "AMXL": "AMX",
    "CHDRAUIB": "CHDRAUI", "GFINBURO": "GFINBUR", "KOFUBL": "KOF",
}

# Alias histórico usado por fetch_intraday
INTRADAY_MAP = BMV_EMISORA_SERIE


def sql_ticker_list() -> list[str]:
    """Lista ordenada de emisora_serie para config/tickers_config.yml."""
    return sorted(set(BMV_EMISORA_SERIE.values()))
