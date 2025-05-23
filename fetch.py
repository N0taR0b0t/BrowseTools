import os
import shelve
import requests
from config import logger

def fetch_from_zyte(url):
    cache_file = "SERP_cache"
    with shelve.open(cache_file) as cache:
        if url in cache:
            print("Loaded SERP from cache.")
            return cache[url]
        else:
            print("Fetching SERP from Zyte API...")
            response = requests.post(
                "https://api.zyte.com/v1/extract",
                auth=(os.getenv("ZYTE_API_KEY"), ""),
                json={"url": url, "serp": True, "serpOptions": {"extractFrom": "browserHtml"}}
            )
            data = response.json().get("serp")
            if data:
                cache[url] = data
                return data

def fetch_browser_html(url):
    CACHE_FILE = "zyte_browser_html_cache"
    with shelve.open(CACHE_FILE) as cache:
        if url in cache:
            logger.info("Loaded browser HTML from cache.")
            return cache[url]
        else:
            logger.info("Fetching browser HTML from Zyte API...")
            response = requests.post(
                "https://api.zyte.com/v1/extract",
                auth=(os.getenv("ZYTE_API_KEY"), ""),
                json={"url": url, "browserHtml": True},
            )
            if response.status_code != 200:
                logger.error(f"Failed to fetch HTML: {response.status_code} - {response.text}")
                return None
            html = response.json().get("browserHtml")
            if html:
                cache[url] = html
                return html