#!/usr/bin/env python3
"""
Universal Precision Extractor v3 — ordered HTML-to-text converter with
optional semantic de-duplication and an embedded JSONL→TXT post-processor.

Key features
------------
* Parses rendered HTML in *document order* (headings, paragraphs, list items).
* Scores blocks heuristically, keeping those above an auto-calibrated threshold
  (70 th percentile, min 0.25 ≤ thr ≤ 0.60).
* Two-stage de-duplication
    1. Fingerprint (exact)      – always on
    2. Sentence-BERT similarity – optional (USE_EMBEDDINGS=True)
* Writes **extracted_content.jsonl** (one block per line) then runs a second
  pass that keeps the *longest* representative of duplicate paragraphs,
  emitting **extracted_content.txt**.

Run:
    python3 extractor.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------#
# Configuration                                                               #
# ---------------------------------------------------------------------------#
HTML_PATH        = Path("browser_html.html")
OUT_TXT          = Path("extracted_content.txt")
OUT_JSONL        = Path("extracted_content.jsonl")

USE_EMBEDDINGS   = False          # set → True to enable Sentence-BERT stage
EMB_MODEL        = "all-MiniLM-L6-v2"

MIN_THRESH       = 0.25
MAX_THRESH       = 0.60
TARGET_PERCENT   = 70             # auto-threshold percentile

BLOCK_TAGS       = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div")

_RE_PUNCT = re.compile(r"[,.!?;:]")
_STOPWORDS = set((
    "the be to of and a in that have i it for not on with he as you do at this "
    "but his by from they we say her she or an will my one all would there "
    "their what so up out if about who get which go me is are was were been "
    "being has had"
).split())

# ---------------------------------------------------------------------------#
# Data structures                                                             #
# ---------------------------------------------------------------------------#
@dataclass
class Block:
    pos: int
    tag: str
    depth: int
    text: str
    score: float
    features: Dict[str, float]
    dup_reason: Optional[str] = None
    embedding: Optional[torch.Tensor] = field(default=None)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("embedding", None)          # not serialisable
        return data

# ---------------------------------------------------------------------------#
# Scoring helpers                                                             #
# ---------------------------------------------------------------------------#
def _collapse_ws(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"\s+([,.!?;:])", r"\1", text)

def _logistic(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    return 1 / (1 + math.exp(-k * (x - x0)))

def _score_text(text: str) -> tuple[float, Dict[str, float]]:
    words = text.split()
    wc = len(words)
    if not wc:
        return 0.0, {"wc": 0, "punc": 0.0, "stop": 0.0}

    punc = len(_RE_PUNCT.findall(text)) / wc
    stop = sum(1 for w in words if w.lower() in _STOPWORDS) / wc

    score = (
        0.50 * _logistic(wc,   k=0.10, x0=40)
        + 0.25 * _logistic(punc, k=10.0, x0=0.05)
        + 0.25 * _logistic(stop, k=10.0, x0=0.25)
    )
    return score, {"wc": wc, "punc": punc, "stop": stop}

def _auto_threshold(blocks: List[Block]) -> float:
    if not blocks:
        return MIN_THRESH
    scores = sorted(b.score for b in blocks)
    thr = scores[int(len(scores) * TARGET_PERCENT / 100)]
    return max(MIN_THRESH, min(MAX_THRESH, thr))

def _fingerprint(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())

# ---------------------------------------------------------------------------#
# Block collection & de-duplication                                           #
# ---------------------------------------------------------------------------#
def collect_blocks(html: str) -> List[Block]:
    soup = BeautifulSoup(html, "lxml")
    blocks: List[Block] = []
    pos = 0

    for node in soup.body.descendants:                 # doc-order traversal
        if isinstance(node, Tag) and node.name in BLOCK_TAGS:
            text = _collapse_ws(node.get_text(" ", strip=True))
            if not text:
                continue
            depth = len(list(node.parents)) - 1
            score, feats = _score_text(text)
            blocks.append(Block(pos, node.name, depth, text, score, feats))
            pos += 1
    return blocks

def dedup_fingerprint(blocks: List[Block]) -> List[Block]:
    seen, keep = set(), []
    for b in blocks:
        fp = _fingerprint(b.text)
        if fp in seen:
            b.dup_reason = "fingerprint"
            continue
        seen.add(fp)
        keep.append(b)
    return keep

def dedup_semantic(blocks: List[Block]) -> List[Block]:
    from sentence_transformers import SentenceTransformer, util   # local import

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, device=device)

    texts = [b.text for b in blocks]
    embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    keep, idxs = [], []
    sim_thr = 0.93 - min(0.08, len(blocks) / 5000)

    for i, b in enumerate(blocks):
        if any(util.pytorch_cos_sim(embs[i], embs[j]).item() >= sim_thr for j in idxs):
            b.dup_reason = "semantic"
            continue
        b.embedding = embs[i]
        keep.append(b)
        idxs.append(i)
    return keep

# ---------------------------------------------------------------------------#
# Embedded jsonl_to_txt functionality (trimmed to essentials)                #
# ---------------------------------------------------------------------------#
_RE_WS   = re.compile(r"\s+")
_RE_PUNC = re.compile(r"[^\w\s]")

def _normalize(text: str) -> str:
    text = _RE_PUNC.sub(" ", text.lower())
    return _RE_WS.sub(" ", text).strip()

def _dedup_keep_longest(lines: List[str]) -> List[str]:
    items = [(i, ln, _normalize(ln)) for i, ln in enumerate(lines)]
    items.sort(key=lambda t: len(t[2]), reverse=True)          # longest first

    kept_raw, canon = {}, []

    for idx, raw, norm in items:
        if any(norm in longer for longer in canon):
            continue
        kept_raw[idx] = raw
        canon.append(norm)

    return [kept_raw[i] for i in sorted(kept_raw)]

def jsonl_to_txt(src: Path, dst: Path) -> None:
    paragraphs: List[str] = []

    with src.open(encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            txt = (rec.get("text") or "").strip()
            if txt:
                paragraphs.append(txt)

    unique = _dedup_keep_longest(paragraphs)

    with dst.open("w", encoding="utf-8") as fout:
        for para in unique:
            fout.write(para + "\n\n")

    print(f"✅ Wrote {len(unique)} unique paragraphs ➜ {dst.resolve()}")

# ---------------------------------------------------------------------------#
# Main                                                                        #
# ---------------------------------------------------------------------------#
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract & de-dup text from HTML/JSONL")
    p.add_argument("--html",  "-i", type=Path, default=HTML_PATH, help="HTML file")
    p.add_argument("--output-jsonl", "-j", type=Path, default=OUT_JSONL,
                   help="JSONL destination")
    p.add_argument("--output-txt",   "-o", type=Path, default=OUT_TXT,
                   help="TXT destination")
    p.add_argument("--embeddings", action="store_true",
                   help="Enable semantic de-duplication")
    return p

def main() -> None:
    args = _build_argparser().parse_args()

    html = args.html.read_text(encoding="utf-8", errors="ignore")
    blocks = collect_blocks(html)

    thr = _auto_threshold(blocks)
    filtered = [b for b in blocks if b.score >= thr]

    dedup = dedup_fingerprint(filtered)
    if args.embeddings or USE_EMBEDDINGS:
        dedup = dedup_semantic(dedup)

    args.output_jsonl.write_text(
        "\n".join(json.dumps(b.to_json(), ensure_ascii=False) for b in dedup),
        encoding="utf-8"
    )

    jsonl_to_txt(args.output_jsonl, args.output_txt)

    print(f"✅ Extracted {len(dedup)} blocks (threshold={thr:.2f}) → "
          f"{args.output_txt.name}, {args.output_jsonl.name}")

if __name__ == "__main__":
    main()