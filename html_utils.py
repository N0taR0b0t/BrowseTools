import re
from bs4 import BeautifulSoup, Tag
from config import DROP_TAGS, SEMANTIC_TAGS, JUNK_CONTAINER_TAGS, BOILERPLATE_PATTERNS, MIN_WORDS

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
        return matches / max(total, 1) > 0.6
    return False

def _jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

def _deduplicate(blocks: list[str], thresh: float = 0.85) -> list[str]:
    unique = []
    for blk in blocks:
        if all(_jaccard(blk, keep) < thresh for keep in unique):
            unique.append(blk)
    return unique

def _extract_segments(parent: Tag) -> list[str]:
    segments = []
    for child in parent.find_all(["p", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"], recursive=False):
        if isinstance(child, Tag):
            txt = _clean_text(child)
            if not _is_trivial(txt):
                segments.append(txt)
    return segments

def visible_text(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(lambda t: t.name in DROP_TAGS):
        el.decompose()

    blocks = []
    for tag in SEMANTIC_TAGS:
        for parent in soup.find_all(tag):
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