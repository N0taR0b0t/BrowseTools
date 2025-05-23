from pathlib import Path
import re
import json
from bs4 import BeautifulSoup, Tag

DROP_TAGS = {"script", "style", "noscript", "template", "svg", "meta", "link"}
SEMANTIC_TAGS = ["main", "article", "section"]
JUNK_CONTAINER_TAGS = {"nav", "header", "footer", "aside", "form"}
BOILERPLATE_PATTERNS = [
    re.compile(r"\bcookie(s| consent)\b", re.I),
    re.compile(r"\bprivacy\b", re.I),
    re.compile(r"\bterms\b", re.I),
    re.compile(r"\bsign\s*in\b", re.I),
]
MIN_WORDS = 15


def _clean_text(el: Tag) -> str:
    txt = el.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\s+([,.!?;:])", r"\1", txt)
    return txt


def _is_trivial(text: str) -> bool:
    if len(text.split()) < MIN_WORDS:
        return True
    if any(p.search(text) for p in BOILERPLATE_PATTERNS):
        total = len(text.split())
        matches = sum(len(p.findall(text)) for p in BOILERPLATE_PATTERNS)
        if matches / max(total, 1) > 0.6:
            return True
    return False


def _jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate(blocks: list[str], thresh: float = 0.85) -> list[str]:
    unique = []
    for blk in blocks:
        if all(_jaccard(blk, keep) < thresh for keep in unique):
            unique.append(blk)
    return unique


def _extract_segments(parent: Tag) -> list[str]:
    segments = []
    for child in parent.find_all(
        ["p", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"], recursive=False
    ):
        if isinstance(child, Tag):
            txt = _clean_text(child)
            if not _is_trivial(txt):
                segments.append(txt)
    return segments


def visible_text(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(lambda t: t.name in DROP_TAGS):
        el.decompose()

    blocks: list[str] = []

    for tag_name in SEMANTIC_TAGS:
        for parent in soup.find_all(tag_name):
            if parent.find_parent(lambda t: t.name in JUNK_CONTAINER_TAGS):
                continue
            blocks.extend(_extract_segments(parent))

    if not blocks:
        for div in soup.find_all("div"):
            if div.find_parent(lambda t: t.name in JUNK_CONTAINER_TAGS):
                continue
            txt = _clean_text(div)
            if not _is_trivial(txt):
                blocks.append(txt)

    return _deduplicate(blocks)


def main():
    html_path = Path("browser_html.html")
    if not html_path.exists():
        print("browser_html.html not found.")
        return

    html_content = html_path.read_text(encoding="utf-8")
    extracted_blocks = visible_text(html_content)

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