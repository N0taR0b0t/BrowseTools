import requests
import shelve
import os
import json
from pathlib import Path

def fetch_from_zyte(url):
    cache_file = "SERP_cache"
    with shelve.open(cache_file) as cache:
        if url in cache:
            print("Loaded from cache.")
            return cache[url]
        else:
            print("Fetching from Zyte API...")
            api_response = requests.post(
                "https://api.zyte.com/v1/extract",
                auth=(os.getenv("ZYTE_API_KEY"), ""),
                json={
                    "url": url,
                    "serp": True,
                    "serpOptions": {"extractFrom": "browserHtml"},
                },
            )
            data = api_response.json().get("serp")
            if data is not None:
                cache[url] = data
            return data

def main():
    url = "https://www.google.com/search?q=tesla+stock+price"
    serp = fetch_from_zyte(url)

    if serp:
        output_path = Path("serp_output.json")
        output_path.write_text(json.dumps(serp, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print("No SERP data returned.")

if __name__ == "__main__":
    main()