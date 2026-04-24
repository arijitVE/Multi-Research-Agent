import random
from datetime import datetime

import arxiv
import requests
import wikipedia
import yfinance as yf
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.tools import tool


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.0 Safari/605.1.15",
]


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information. Returns titles, URLs and snippets."""
    try:
        results = DDGS().text(query, max_results=5, timelimit="m")
        out = []
        for result in results:
            out.append(
                "\n".join(
                    [
                        f"Title: {result.get('title', 'N/A')}",
                        f"URL: {result.get('href', 'N/A')}",
                        f"Snippet: {result.get('body', 'N/A')}",
                    ]
                )
            )
        return "\n-----\n".join(out) if out else "No web search results found."
    except Exception as e:
        return f"Search failed: {e}"


@tool
def news_search(query: str) -> str:
    """Search for recent news on a topic. Use for current events, prices, and breaking news."""
    try:
        results = DDGS().news(query, max_results=5, timelimit="d")
        out = []
        for result in results:
            out.append(
                "\n".join(
                    [
                        f"Headline: {result.get('title', 'N/A')}",
                        f"Source: {result.get('source', 'N/A')}",
                        f"URL: {result.get('url', 'N/A')}",
                        f"Published: {result.get('date', 'N/A')}",
                        f"Summary: {result.get('body', 'N/A')}",
                    ]
                )
            )
        return "\n-----\n".join(out) if out else "No recent news results found."
    except Exception as e:
        return f"News search failed: {e}"


@tool
def wikipedia_search(query: str) -> str:
    """Get factual background information from Wikipedia. Use for definitions, history, and context."""
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return f"Source: Wikipedia\n\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=True)
            return f"Source: Wikipedia\n\n{summary}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for: {query}"
        except Exception as inner_e:
            return f"Wikipedia lookup failed: {inner_e}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for: {query}"
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


@tool
def arxiv_search(query: str) -> str:
    """Search academic papers on arXiv. Use for scientific research, ML, physics, CS topics."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=4,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out = []
        for result in client.results(search):
            authors = ", ".join(author.name for author in result.authors[:3])
            published = result.published.date().isoformat()
            summary = (result.summary or "").replace("\n", " ")[:400]
            out.append(
                "\n".join(
                    [
                        f"Title: {result.title}",
                        f"Authors: {authors}",
                        f"Published: {published}",
                        f"Summary: {summary}",
                        f"URL: {result.entry_id}",
                    ]
                )
            )
        return "\n-----\n".join(out) if out else "No arXiv papers found."
    except Exception as e:
        return f"arXiv search failed: {e}"


@tool
def financial_data(ticker: str) -> str:
    """Get live financial data for a stock, ETF, currency or commodity.
    Use ticker symbols: AAPL for Apple, GC=F for gold, BTC-USD for Bitcoin, EURUSD=X for forex."""
    try:
        asset = yf.Ticker(ticker)
        info = asset.info or {}
        history = asset.history(period="5d")

        name = info.get("longName") or info.get("shortName") or ticker
        price = info.get("currentPrice")
        if price is None:
            price = info.get("regularMarketPrice")
        if price is None and not history.empty:
            price = history["Close"].dropna().iloc[-1]

        currency = info.get("currency") or "N/A"
        low = info.get("fiftyTwoWeekLow")
        high = info.get("fiftyTwoWeekHigh")
        market_cap = info.get("marketCap")
        sector = info.get("sector")

        lines = [
            f"Asset: {name} ({ticker})",
            f"Current price: {price} {currency}",
            f"52-week range: {low} - {high}",
        ]
        if market_cap is not None:
            lines.append(f"Market cap: {market_cap}")
        if sector:
            lines.append(f"Sector: {sector}")
        lines.append(f"Data as of: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        return "\n".join(lines)
    except Exception as e:
        return f"Financial data lookup failed for {ticker}: {e}"


@tool
def weather_search(location: str) -> str:
    """Get current weather and 3-day forecast for any city or location."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=15,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        results = geo_data.get("results") or []
        if not results:
            return f"Location not found: {location}"

        place = results[0]
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": [
                    "temperature_2m",
                    "weathercode",
                    "wind_speed_10m",
                    "relative_humidity_2m",
                ],
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                ],
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current", {})
        daily = weather_data.get("daily", {})
        maxes = daily.get("temperature_2m_max", [])
        mins = daily.get("temperature_2m_min", [])

        forecast_parts = []
        for index in range(min(3, len(maxes), len(mins))):
            forecast_parts.append(f"Day {index + 1}: {maxes[index]} C / {mins[index]} C")

        return "\n".join(
            [
                f"Location: {place.get('name')}, {place.get('country')}",
                (
                    "Current: "
                    f"{current.get('temperature_2m')} C, "
                    f"wind {current.get('wind_speed_10m')} km/h, "
                    f"humidity {current.get('relative_humidity_2m')}%"
                ),
                f"Forecast (3 days): {', '.join(forecast_parts)}",
            ]
        )
    except Exception as e:
        return f"Weather lookup failed for {location}: {e}"


def _scrape_with_jina(url: str) -> str | None:
    jina_url = f"https://r.jina.ai/{url}"
    response = requests.get(
        jina_url,
        headers={"Accept": "text/plain", "X-Return-Format": "text"},
        timeout=15,
    )
    if response.status_code == 200 and len(response.text.strip()) > 200:
        print(f"[web_scrape] Tier 1 succeeded for {url}")
        return response.text[:4000]
    return None


def _scrape_with_playwright(url: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.wait_for_load_state("networkidle", timeout=10000)
            text = page.inner_text("body")[:4000]
            if len(text.strip()) > 200:
                print(f"[web_scrape] Tier 2 succeeded for {url}")
                return text
    finally:
        if browser is not None:
            browser.close()
    return None


def _scrape_with_requests(url: str) -> str | None:
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": random.choice(USER_AGENTS)},
    )
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)[:4000]
    if len(text.strip()) > 100:
        print(f"[web_scrape] Tier 3 succeeded for {url}")
        return text
    return None


@tool
def web_scrape(url: str) -> str:
    """Scrape and return clean text content from a URL. Handles JS-heavy pages automatically."""
    for scraper in (_scrape_with_jina, _scrape_with_playwright, _scrape_with_requests):
        try:
            text = scraper(url)
            if text:
                return text
        except Exception:
            continue
    return f"Unable to scrape {url}: all methods failed."
