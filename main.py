from visitSite import fetch_browser_html  # from visitSite.py
from visible import visible_text             # from visible.py
import json
import re
from pathlib import Path

def main():
    url = "https://finance.yahoo.com/quote/TSLA/"
    browser_html = fetch_browser_html(url)

    if not browser_html:
        print("Failed to retrieve HTML.")
        return

    extracted_blocks = visible_text(browser_html)

    word_limit = 650
    words_printed = 0
    output_blocks = []

    for block in extracted_blocks:
        block_words = block.split()
        remaining = word_limit - words_printed

        if remaining <= 0:
            break

        if len(block_words) > remaining:
            truncated = " ".join(block_words[:remaining + 30])
            period_match = list(re.finditer(r"\. ", truncated))
            if period_match:
                last_period_idx = period_match[-1].end()
                trimmed = truncated[:last_period_idx].strip()
                trimmed_words = trimmed.split()
                output_blocks.append(trimmed)
                words_printed += len(trimmed_words)
            else:
                chunk = " ".join(block_words[:remaining])
                output_blocks.append(chunk)
                words_printed += remaining
            break
        else:
            output_blocks.append(block)
            words_printed += len(block_words)

    output_path = Path("visible_text.json")
    output_path.write_text(json.dumps(output_blocks, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()