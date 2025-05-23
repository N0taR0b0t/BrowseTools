import requests
import shelve
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

CACHE_FILE = "zyte_browser_html_cache"

def fetch_browser_html(url):
    """Fetch browser-rendered HTML via Zyte API with caching."""
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

            browser_html = response.json().get("browserHtml")
            if browser_html is not None:
                cache[url] = browser_html
            return browser_html