from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re
import math

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("SmartSearch requires pandas installed.") from e


# Simple alias map — extend as you wish.
CANONICAL_TAGS = {
    "beach": {"sea", "seaside", "ocean", "shore", "coast", "beachfront", "beachside"},
    "mountain": {"alps", "alpine", "peak", "summit", "mountains", "hill", "hillside"},
    "city": {"downtown", "urban", "central", "metropolitan", "metro"},
    "cabin": {"chalet", "lodge", "hut", "cottage"},
    "apartment": {"flat", "condo", "suite"},
    "pool": {"swimming", "swimmingpool", "plunge"},
    "hot tub": {"jacuzzi", "spa"},
    "wifi": {"wi-fi", "internet"},
}

def normalize_token(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def tokenize(s: str) -> List[str]:
    s = normalize_token(s)
    return [w for w in s.split(" ") if w]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def alias_expand(tokens: List[str]) -> List[str]:
    # Map aliases back to canonical keys when hit
    expanded = set(tokens)
    for canon, aliases in CANONICAL_TAGS.items():
        if canon in expanded or any(a in expanded for a in aliases):
            expanded.add(canon)
            expanded |= set(aliases)
    return list(expanded)

class SmartSearch:
    def __init__(self, properties_df: "pd.DataFrame"):
        # Build a lightweight corpus from your CSV fields
        df = properties_df.copy()
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        parts = []
        for col in ["location", "type", "features", "tags", "description"]:
            if col in df.columns:
                parts.append(df[col].fillna("").astype(str))
        if parts:
            full = parts[0]
            for p in parts[1:]:
                full = full + " | " + p
        else:
            # Fallback to all columns
            full = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1)

        df["__fulltext__"] = full.apply(normalize_token)
        self.df = df

    def find_candidates(self, query: str, top_k: int = 50) -> "pd.DataFrame":
        q_tokens = alias_expand(tokenize(query))
        # Score each row by Jaccard over token sets
        scores = []
        for i, text in enumerate(self.df["__fulltext__"].tolist()):
            t_tokens = alias_expand(tokenize(text))
            s = jaccard(q_tokens, t_tokens)
            scores.append(s)
        scored = self.df.copy()
        scored["semantic_score"] = scores
        scored = scored.sort_values("semantic_score", ascending=False).head(top_k)
        return scored

    def canonicalize_tags(self, raw_tags: List[str]) -> List[str]:
        out = set()
        for t in raw_tags:
            t = normalize_token(t)
            out.add(t)
            for canon, aliases in CANONICAL_TAGS.items():
                if t == canon or t in aliases:
                    out.add(canon)
                    out |= aliases
        return sorted(out)
