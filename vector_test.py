"""
vector_test.py — rock-solid local Chroma demo
Compatible with:
  torch==2.2.2
  transformers==4.38.1
  sentence-transformers==2.6.1
  chromadb==0.4.24
  numpy<2.0.0
"""

from __future__ import annotations
from typing import List
from pathlib import Path
import warnings, sys, importlib

# ──────────────────────────────────────────────────────────────
# 0️⃣  Safety & Version Gate
# ──────────────────────────────────────────────────────────────
REQUIRED = {
    "torch": "2.2.2",
    "transformers": "4.38.1",
    "sentence_transformers": "2.6.1",
    "chromadb": "0.4.24",
    "numpy_max_major": 1,          # numpy must be 1.x
}

def require(pkg: str, exact: str):
    mod = importlib.import_module(pkg)
    if mod.__version__ != exact:
        sys.exit(
            f"[ERROR] {pkg} version must be {exact}, found {mod.__version__}. "
            f"Run:  pip install '{pkg}=={exact}'"
        )

require("torch",                  REQUIRED["torch"])
require("transformers",           REQUIRED["transformers"])
require("sentence_transformers",  REQUIRED["sentence_transformers"])
require("chromadb",               REQUIRED["chromadb"])

import numpy as _np
if int(_np.__version__.split(".")[0]) > REQUIRED["numpy_max_major"]:
    sys.exit(
        f"[ERROR] numpy must be <2.0.0, found {_np.__version__}. "
        "Run:  pip install 'numpy<2.0.0'"
    )

warnings.filterwarnings("ignore")  # silence HF & tokenizers chatter

# ──────────────────────────────────────────────────────────────
# 1️⃣  Define a Chroma-compliant embedding function
# ──────────────────────────────────────────────────────────────
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
import torch

class SBERTEmbedding(EmbeddingFunction):
    """
    Minimal, CPU-safe embedding function that satisfies
    Chroma’s 0.4.16+ interface:
      __call__(input: List[str]) -> List[List[float]]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Force CPU – avoids any default-device gymnastics
        self.device = torch.device("cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))

    def __call__(self, input: List[str]) -> List[List[float]]:          # noqa: D401
        """Return dense vectors for a list of texts."""
        with torch.no_grad():
            vectors = self.model.encode(
                input,
                device=self.device,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vectors.tolist()

# ──────────────────────────────────────────────────────────────
# 2️⃣  Chroma client & collection
# ──────────────────────────────────────────────────────────────
import chromadb

DB_PATH = Path("./chroma_test_db").resolve()
client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="investment_news",
    embedding_function=SBERTEmbedding(),   # attach our function
)

# ──────────────────────────────────────────────────────────────
# 3️⃣  Seed documents (only once)
# ──────────────────────────────────────────────────────────────
if collection.count() == 0:
    collection.add(
        documents=[
            "Nvidia stock surged 10% after the earnings report.",
            "Tesla's market cap continues to decline amid growing EV competition.",
            "The Federal Reserve hinted at future interest rate hikes in 2025.",
        ],
        metadatas=[
            {"source": "TechCrunch"},
            {"source": "Reuters"},
            {"source": "Bloomberg"},
        ],
        ids=["doc1", "doc2", "doc3"],
    )

# ──────────────────────────────────────────────────────────────
# 4️⃣  Query demo
# ──────────────────────────────────────────────────────────────
QUERY = "AI hardware companies earnings"

results = collection.query(query_texts=[QUERY], n_results=3)

print(f"\n🔎  Query: {QUERY}")
for rank, (doc, meta, dist) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ),
    start=1,
):
    print(
        f"\nRank {rank}\n"
        f"Document : {doc}\n"
        f"Metadata : {meta}\n"
        f"Distance : {dist:.4f}"
    )