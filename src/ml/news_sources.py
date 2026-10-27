"""
Unified news source fetchers — dict schema compatible with news.Article conversion.

Each fetcher returns list[dict] with keys:
  title, summary, url, published_ts, source, relevance_score,
  av_sentiment (optional, Alpha Vantage).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # Windows: sin flock, el lock queda a cargo de _gdelt_wait_turn
    fcntl = None

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10
GDELT_REQUEST_TIMEOUT = 45
MAX_ARTICLES = 30
RATE_LIMIT_SLEEP = 0.3
GDELT_MIN_INTERVAL_S = float(os.environ.get("GDELT_INTERVAL_S", "60"))
GDELT_MAX_RETRIES = 3
GDELT_429_COOLDOWN_S = 180.0
GDELT_LOCK_PATH = Path(__file__).resolve().parents[2] / "data" / "news" / "history" / ".gdelt.lock"
GDELT_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "news" / "history" / ".gdelt_state.json"

# ── Entity aliases: ticker → search terms for relevance beyond literal ticker ──

ENTITY_ALIASES: dict[str, list[str]] = {
    "ALSEA": ["Alsea", "ALSEA", "restaurantes"],
    "WALMEX": ["Walmex", "Walmart Mexico", "WALMEX"],
    "FEMSAUBD": ["FEMSA", "Coca-Cola", "FEMSAUBD"],
    "GMEXICOB": ["Grupo Mexico", "GMEXICOB", "mineria"],
    "CEMEXCPO": ["Cemex", "CEMEXCPO", "cemento"],
    "GFNORTEO": ["GFNorte", "Banorte", "GFNORTEO", "Grupo Financiero Banorte"],
    "BIMBOA": ["Bimbo", "BIMBOA"],
    "KIMBERA": ["Kimberly Clark", "Kimber", "KIMBERA"],
    "PINFRA": ["Pinfra", "PINFRA", "autopistas"],
    "ORBIA": ["Orbia", "Mexichem", "ORBIA"],
    "GRUMAB": ["Gruma", "GRUMAB", "tortilla"],
    "ASURB": ["Asur", "ASURB", "aeropuertos sureste"],
    "OMAB": ["OMA", "OMAB", "aeropuertos"],
    "GAPB": ["GAP", "GAPB", "aeropuertos pacifico"],
    "GCARSOA1": ["GCarso", "Carso", "GCARSOA1", "Slim"],
    "MEGACPO": ["Megacable", "MEGACPO"],
    "LABB": ["Genomma Lab", "LABB", "Genomma"],
    "CUERVO": ["Jose Cuervo", "Cuervo", "CUERVO", "tequila"],
    "AC": ["Arca Continental", "Arca", "AC"],
    "VESTA": ["Vesta", "VESTA", "parques industriales"],
    "BOLSAA": ["Bolsa Mexicana", "BMV", "BOLSAA"],
    "GCC": ["GCC", "cemento"],
    "HOTEL": ["City Express", "Hoteles City", "HOTEL"],
    "POSADAS": ["Posadas", "POSADAS", "hoteles"],
    "TLEVISA": ["Televisa", "TLEVISA"],
    "HERDEZ": ["Herdez", "HERDEZ"],
    "GENTERA": ["Gentera", "GENTERA", "microfinanzas"],
    "FUNO11": ["Fibra Uno", "FUNO", "FUNO11"],
    "VISTA": ["Vista Energy", "VISTA"],
    "VOLARA": ["Volaris", "VOLARA"],
    "Q": ["Qualitas", "Quálitas", "Q", "seguros"],
    "AAPL": ["Apple", "AAPL", "iPhone"],
    "MSFT": ["Microsoft", "MSFT", "Azure"],
    "NVDA": ["NVIDIA", "NVDA", "GPU"],
    "GOOGL": ["Alphabet", "Google", "GOOGL"],
    "META": ["Meta", "Facebook", "META"],
    "AMZN": ["Amazon", "AMZN", "AWS"],
    "TSLA": ["Tesla", "TSLA"],
    "AMD": ["AMD", "semiconductor"],
    "INTC": ["Intel", "INTC"],
    "JPM": ["JPMorgan", "JPM", "Chase"],
    "BAC": ["Bank of America", "BAC"],
    "GS": ["Goldman Sachs", "GS"],
    "V": ["Visa", "V"],
    "MA": ["Mastercard", "MA"],
    "KO": ["Coca-Cola", "KO"],
    "PEP": ["PepsiCo", "PEP"],
    "MCD": ["McDonald", "MCD"],
    "JNJ": ["Johnson", "JNJ", "Johnson & Johnson"],
    "PFE": ["Pfizer", "PFE"],
    "XOM": ["ExxonMobil", "XOM"],
    "CVX": ["Chevron", "CVX"],
    "AMX": ["America Movil", "AMX", "Telcel"],
    "FMX": ["FEMSA", "FMX"],
    "SPY": ["S&P 500", "SPY", "SP500"],
    "QQQ": ["NASDAQ", "QQQ"],
    "GLD": ["gold", "GLD"],
    "SLV": ["silver", "SLV"],
    "TLT": ["treasury", "TLT", "bonds"],
    "EEM": ["emerging markets", "EEM"],
    "EWZ": ["Brazil", "EWZ"],
    "FXI": ["China", "FXI"],
    "XLE": ["energy ETF", "XLE", "oil sector"],
    "XLF": ["financial ETF", "XLF"],
    "XLK": ["technology ETF", "XLK", "tech sector"],
    "IBIT": ["Bitcoin", "IBIT", "BlackRock"],
    "FBTC": ["Bitcoin", "FBTC", "Fidelity"],
    "ETHA": ["Ethereum", "ETHA", "iShares"],
    "GBTC": ["Bitcoin", "GBTC", "Grayscale"],
    "MSTR": ["MicroStrategy", "MSTR", "Bitcoin"],
    "COIN": ["Coinbase", "COIN", "crypto exchange"],
    "MARA": ["Marathon Digital", "MARA", "Bitcoin mining"],
    "RIOT": ["Riot Platforms", "RIOT", "Bitcoin mining"],
    "CLSK": ["CleanSpark", "CLSK", "Bitcoin mining"],
    "FIBRAPL": ["Fibra Prologis", "Prologis", "FIBRAPL"],
    "FINN13": ["Fibra Inn", "FINN13"],
}

# Términos sectoriales para cables DataBursátil (Indexity / mercado general)
SECTOR_ALIASES: dict[str, list[str]] = {
    "GFNORTEO": ["banca", "bancario", "bancos", "banorte", "crédito"],
    "BIMBOA": ["panificadora", "consumo", "alimentos"],
    "FEMSAUBD": ["bebidas", "refrescos", "oxxo"],
    "CEMEXCPO": ["cemento", "construcción", "construccion"],
    "GMEXICOB": ["minería", "mineria", "cobre", "minera"],
    "WALMEX": ["retail", "autoservicio", "walmart"],
    "ALSEA": ["restaurantes", "consumo"],
    "GFINBURO": ["banca", "financiero", "inbursa"],
    "VOLARA": ["aerolínea", "aerolinea", "aviación", "aviacion"],
    "ASURB": ["aeropuerto", "turismo"],
    "OMAB": ["aeropuerto", "oma"],
    "GAPB": ["aeropuerto", "gap"],
    "TLEVISA": ["televisión", "television", "media"],
    "BOLSAA": ["bolsa", "bmv", "ipc"],
    "Q": ["seguros", "qualitas"],
    "HERDEZ": ["alimentos", "herdez", "salsas"],
    "HOTEL": ["hotelería", "hoteleria", "hoteles", "city express"],
    "VESTA": ["fibras", "industrial", "logística", "logistica"],
    "FUNO11": ["fibra", "fibras", "reit"],
}

_CRYPTO_SYMBOL_MAP = {
    "IBIT": "BTC", "FBTC": "BTC", "GBTC": "BTC",
    "ETHA": "ETH",
    "MSTR": "BTC", "COIN": "COIN", "MARA": "BTC", "RIOT": "BTC", "CLSK": "BTC",
}

_ESG_KEYWORDS = frozenset({
    "esg", "sustainability", "sustainable", "carbon", "climate", "renewable",
    "green", "emissions", "net zero", "net-zero", "csr", "social responsibility",
    "environmental", "governance", "diversity", "inclusion", "dei",
    "sostenibilidad", "sostenible", "medio ambiente", "ambiental", "climático",
    "climatico", "renovable", "emisiones", "carbono", "responsabilidad social",
    "gobernanza", "diversidad", "inclusión", "inclusion",
})


def get_entity_aliases(ticker: str, extra: Optional[list[str]] = None) -> list[str]:
    """Return deduplicated search terms for relevance matching."""
    seen: set[str] = set()
    out: list[str] = []
    combined = (
        [ticker]
        + ENTITY_ALIASES.get(ticker, [])
        + SECTOR_ALIASES.get(ticker, [])
        + (extra or [])
    )
    for term in combined:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(term.strip())
    return out


def compute_relevance_score(title: str, summary: str, aliases: list[str]) -> float:
    """
    Score 0–1 how relevant an article is to the entity aliases.
    Title matches weigh more than summary-only matches.
    """
    if not title and not summary:
        return 0.0

    title_l = title.lower()
    text_l = f"{title} {summary}".lower()
    best = 0.0

    for alias in aliases:
        a = alias.strip().lower()
        if not a or len(a) < 2:
            continue
        if a in title_l:
            best = max(best, 1.0)
        elif a in text_l:
            best = max(best, 0.75)
        elif len(a) >= 4 and re.search(rf"\b{re.escape(a)}\b", text_l):
            best = max(best, 0.55)

    # DataBursátil Indexity: "Indexity - OMA - titular" o "17:24 Indexity - WALMEX* - ..."
    idx = re.search(r"indexity\s*-\s*([\w*]+)\s*-", title_l)
    if idx:
        code = idx.group(1).replace("*", "").lower()
        alias_codes = {a.replace("*", "").lower() for a in aliases}
        if code in alias_codes:
            best = max(best, 1.0)
        else:
            for ac in alias_codes:
                if len(code) >= 2 and (ac.startswith(code) or code.startswith(ac[: min(3, len(ac))])):
                    best = max(best, 1.0)
                    break

    if best == 0.0 and text_l.strip():
        return 0.15
    return round(min(best, 1.0), 3)


def is_esg_headline(title: str) -> bool:
    """True if headline appears to be ESG / sustainability related."""
    lower = title.lower()
    return any(kw in lower for kw in _ESG_KEYWORDS)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def _article(
    *,
    title: str,
    summary: str = "",
    url: str = "",
    published_ts: int = 0,
    source: str,
    relevance_score: float = 1.0,
    av_sentiment: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "summary": summary,
        "url": url,
        "published_ts": published_ts,
        "source": source,
        "relevance_score": relevance_score,
        "av_sentiment": av_sentiment,
    }


def _with_relevance(
    raw: dict[str, Any],
    ticker: str,
    aliases: list[str],
) -> dict[str, Any]:
    score = compute_relevance_score(
        raw.get("title", ""),
        raw.get("summary", ""),
        aliases or get_entity_aliases(ticker),
    )
    raw["relevance_score"] = score
    return raw


def _parse_rss_items(
    url: str,
    source: str,
    max_items: int,
    keyword_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        articles: list[dict[str, Any]] = []

        for item in root.iter("item"):
            title = item.findtext("title", "")
            summary = item.findtext("description", "") or item.findtext("summary", "")
            link = item.findtext("link", "")
            pub_raw = item.findtext("pubDate", "")

            if keyword_filter and keyword_filter.lower() not in (title + summary).lower():
                continue

            ts = 0
            if pub_raw:
                try:
                    ts = int(parsedate_to_datetime(pub_raw).timestamp())
                except Exception:
                    pass

            if title:
                articles.append(_article(
                    title=title,
                    summary=_strip_html(summary or ""),
                    url=link,
                    published_ts=ts,
                    source=source,
                ))
            if len(articles) >= max_items:
                break
        return articles
    except Exception as e:
        logger.debug(f"RSS {url}: {e}")
        return []


def _parse_databursatil_date(fecha: str, title: str = "") -> int:
    """Parse fecha field or HH:MM prefix in cable title (DataBursátil usa ``t`` sin fecha)."""
    if fecha:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(fecha[:19], fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except ValueError:
                continue
    m = re.match(r"^(\d{1,2}):(\d{2})\s", title or "")
    if m:
        now = datetime.now(tz=timezone.utc)
        dt = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        return int(dt.timestamp())
    return int(datetime.now(tz=timezone.utc).timestamp())


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_databursatil_articles(
    ticker: str,
    aliases: Optional[list[str]] = None,
    max_items: int = MAX_ARTICLES,
) -> list[dict[str, Any]]:
    """DataBursátil V2 — noticias + cables (requiere DATABURSATIL_API_TOKEN)."""
    if not os.getenv("DATABURSATIL_API_TOKEN"):
        return []
    try:
        from src.ml.databursatil_news import fetch_databursatil_all_articles
        return fetch_databursatil_all_articles(ticker, aliases, max_items=max_items)
    except Exception as e:
        logger.debug(f"DataBursátil {ticker}: {e}")
        return []


def fetch_alpha_vantage_articles(
    ticker: str,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Alpha Vantage NEWS_SENTIMENT — preserves av_sentiment scores."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return []

    alias_list = aliases or get_entity_aliases(ticker)
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "apikey": api_key,
                "limit": max_items,
                "sort": "LATEST",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if not resp.ok:
            return []

        feed = resp.json().get("feed", [])
        out: list[dict[str, Any]] = []
        for item in feed[:max_items]:
            title = item.get("title", "")
            if not title:
                continue

            av_score: Optional[float] = None
            raw_score = item.get("overall_sentiment_score")
            if raw_score is not None and raw_score != "":
                try:
                    av_score = float(raw_score)
                except (ValueError, TypeError):
                    pass

            ts = 0
            time_pub = item.get("time_published", "")
            if time_pub:
                for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d%H%M%S"):
                    try:
                        dt = datetime.strptime(time_pub[:15], fmt).replace(tzinfo=timezone.utc)
                        ts = int(dt.timestamp())
                        break
                    except ValueError:
                        continue

            art = _article(
                title=title,
                summary=item.get("summary", "") or "",
                url=item.get("url", ""),
                published_ts=ts,
                source="alpha_vantage",
                av_sentiment=av_score,
            )
            out.append(_with_relevance(art, ticker, alias_list))
        return out
    except Exception as e:
        logger.debug(f"Alpha Vantage {ticker}: {e}")
        return []


def fetch_yfinance_articles(
    yf_symbol: str,
    ticker: str,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    alias_list = aliases or get_entity_aliases(ticker)
    try:
        items = yf.Ticker(yf_symbol).news or []
        out: list[dict[str, Any]] = []
        for item in items[:max_items]:
            ts = item.get("providerPublishTime") or item.get("publisher_time", 0)
            art = _article(
                title=item.get("title", ""),
                summary=item.get("summary", "") or item.get("description", ""),
                url=item.get("link", "") or item.get("url", ""),
                published_ts=int(ts) if ts else 0,
                source="yfinance",
            )
            if art["published_ts"] > 0:
                out.append(_with_relevance(art, ticker, alias_list))
        return out
    except Exception as e:
        logger.debug(f"yfinance news {yf_symbol}: {e}")
        return []


def fetch_google_rss(
    query: str,
    ticker: str,
    max_items: int = MAX_ARTICLES,
    lang: str = "es-MX",
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    alias_list = aliases or get_entity_aliases(ticker)
    encoded = requests.utils.quote(query)
    region = "MX" if "MX" in lang.upper() else "US"
    ceid = f"{region}:{lang.split('-')[0]}"
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl={lang}&gl={region}&ceid={ceid}"
    )
    time.sleep(RATE_LIMIT_SLEEP)
    raw = _parse_rss_items(url, "google_news", max_items)
    return [_with_relevance(a, ticker, alias_list) for a in raw]


def fetch_elfinanciero(
    ticker: str,
    keyword: str,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    alias_list = aliases or get_entity_aliases(ticker)
    feeds = [
        "https://www.elfinanciero.com.mx/rss/mercados.xml",
        "https://www.elfinanciero.com.mx/rss/economia.xml",
    ]
    articles: list[dict[str, Any]] = []
    for feed_url in feeds:
        articles.extend(_parse_rss_items(feed_url, "elfinanciero", max_items, keyword_filter=keyword))
        time.sleep(RATE_LIMIT_SLEEP)
    return [_with_relevance(a, ticker, alias_list) for a in articles[:max_items]]


def fetch_expansion(
    ticker: str,
    keyword: str,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    alias_list = aliases or get_entity_aliases(ticker)
    feeds = [
        "https://expansion.mx/rss/empresas",
        "https://expansion.mx/rss/mercados",
    ]
    articles: list[dict[str, Any]] = []
    for feed_url in feeds:
        articles.extend(_parse_rss_items(feed_url, "expansion", max_items, keyword_filter=keyword))
        time.sleep(RATE_LIMIT_SLEEP)
    return [_with_relevance(a, ticker, alias_list) for a in articles[:max_items]]


@contextmanager
def gdelt_request_lock():
    """Un solo request GDELT a la vez (evita 429 por procesos paralelos)."""
    GDELT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GDELT_LOCK_PATH, "w") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        _gdelt_wait_turn()
        try:
            yield
        finally:
            pass


def configure_gdelt_interval(seconds: float) -> None:
    """Ajusta intervalo mínimo entre requests GDELT (CLI)."""
    global GDELT_MIN_INTERVAL_S
    GDELT_MIN_INTERVAL_S = max(30.0, float(seconds))


def _gdelt_read_state() -> dict:
    if not GDELT_STATE_PATH.exists():
        return {}
    try:
        return json.loads(GDELT_STATE_PATH.read_text())
    except Exception:
        return {}


def _gdelt_write_state(state: dict) -> None:
    GDELT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GDELT_STATE_PATH.write_text(json.dumps(state))


def _gdelt_wait_turn() -> None:
    """Espera cooldown global + intervalo mínimo (persiste entre procesos)."""
    state = _gdelt_read_state()
    now = time.time()
    cooldown_until = float(state.get("cooldown_until") or 0)
    last_req = float(state.get("last_request") or 0)
    wait_until = max(cooldown_until, last_req + GDELT_MIN_INTERVAL_S)
    delay = wait_until - now
    if delay > 0:
        logger.info("GDELT throttle — esperando %.0fs", delay)
        time.sleep(delay)


def _gdelt_mark_request(*, cooldown_s: float = 0) -> None:
    now = time.time()
    state = _gdelt_read_state()
    state["last_request"] = now
    if cooldown_s > 0:
        state["cooldown_until"] = now + cooldown_s
    elif float(state.get("cooldown_until") or 0) <= now:
        state.pop("cooldown_until", None)
    _gdelt_write_state(state)


def _gdelt_get(url: str) -> requests.Response | None:
    """GET GDELT — un intento; 429 activa cooldown global sin retries encadenados."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; stock-market/1.0)"}
    with gdelt_request_lock():
        try:
            resp = requests.get(url, timeout=GDELT_REQUEST_TIMEOUT, headers=headers)
        except requests.RequestException as exc:
            logger.warning("GDELT error: %s", exc)
            _gdelt_mark_request(cooldown_s=GDELT_MIN_INTERVAL_S)
            return None
        _gdelt_mark_request()
        if resp.status_code == 429:
            logger.warning("GDELT 429 — cooldown %.0fs", GDELT_429_COOLDOWN_S)
            _gdelt_mark_request(cooldown_s=GDELT_429_COOLDOWN_S)
            return None
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            logger.warning("GDELT HTTP %s", resp.status_code)
            return None
        return resp


def fetch_gdelt(
    query: str,
    ticker: str,
    days_back: int = 7,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    alias_list = aliases or get_entity_aliases(ticker)
    try:
        encoded = requests.utils.quote(query)
        url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={encoded}&mode=artlist&maxrecords={max_items}"
            f"&format=json&timespan={days_back}d&sourcelang=spanish+english"
        )
        resp = _gdelt_get(url)
        if resp is None:
            return []
        articles_raw = resp.json().get("articles") or []
        out: list[dict[str, Any]] = []
        for art in articles_raw:
            title = art.get("title", "")
            seen = art.get("seendate", "")
            ts = 0
            if seen:
                try:
                    dt = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    ts = int(dt.timestamp())
                except ValueError:
                    pass
            if title:
                item = _article(
                    title=title, summary="", url=art.get("url", ""),
                    published_ts=ts, source="gdelt",
                )
                out.append(_with_relevance(item, ticker, alias_list))
        return out
    except Exception as e:
        logger.debug(f"GDELT {query}: {e}")
        return []


def fetch_gdelt_range(
    query: str,
    ticker: str,
    start: datetime,
    end: datetime,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], bool]:
    """GDELT doc API — rango de fechas. Retorna (artículos, ok). ok=False → no cachear."""
    alias_list = aliases or get_entity_aliases(ticker)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    try:
        encoded = requests.utils.quote(query)
        start_s = start.strftime("%Y%m%d%H%M%S")
        end_s = end.strftime("%Y%m%d%H%M%S")
        url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={encoded}&mode=artlist&maxrecords={max_items}"
            f"&format=json&startdatetime={start_s}&enddatetime={end_s}"
            f"&sourcelang=spanish+english"
        )
        resp = _gdelt_get(url)
        if resp is None:
            return [], False
        articles_raw = resp.json().get("articles") or []
        out: list[dict[str, Any]] = []
        for art in articles_raw:
            title = art.get("title", "")
            seen = art.get("seendate", "")
            ts = 0
            if seen:
                try:
                    dt = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    ts = int(dt.timestamp())
                except ValueError:
                    pass
            if title:
                item = _article(
                    title=title, summary="", url=art.get("url", ""),
                    published_ts=ts, source="gdelt",
                )
                out.append(_with_relevance(item, ticker, alias_list))
        return out, True
    except Exception as e:
        logger.debug(f"GDELT range {query} {start.date()}-{end.date()}: {e}")
        return [], False


def fetch_cryptopanic(
    ticker: str,
    max_items: int = MAX_ARTICLES,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("CRYPTOPANIC_API_KEY")
    if not api_key:
        return []

    alias_list = aliases or get_entity_aliases(ticker)
    currency = _CRYPTO_SYMBOL_MAP.get(ticker, "BTC")
    try:
        resp = requests.get(
            f"https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": api_key, "currencies": currency, "public": "true"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        out: list[dict[str, Any]] = []
        for post in (resp.json().get("results") or [])[:max_items]:
            ts = 0
            pub = post.get("published_at", "")
            if pub:
                try:
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    ts = int(dt.timestamp())
                except ValueError:
                    pass
            art = _article(
                title=post.get("title", ""),
                summary=post.get("body") or "",
                url=post.get("url", ""),
                published_ts=ts,
                source="cryptopanic",
            )
            out.append(_with_relevance(art, ticker, alias_list))
        return out
    except Exception as e:
        logger.debug(f"CryptoPanic {currency}: {e}")
        return []
