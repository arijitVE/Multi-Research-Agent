from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import  TavilyClient
import os
from dotenv import load_dotenv
from rich import print
import random
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")
tavily = TavilyClient(api_key=api_key)

@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic , Returns Titles , URLs and snippts"""
    results = tavily.search(query=query , max_results= 5)

    out = []

    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    return "\n-----\n".join(out)


@tool
def web_scrape(url : str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    USER_AGENTS = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.0 Safari/605.1.15",]
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": random.choice(USER_AGENTS)})
        # if response.status_code != 200:
        #     return f"Failed to fetch page: {response.status_code}"
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True) [:3000]
    except Exception as e:
        return f"Error scraping the web page: {str(e)}"