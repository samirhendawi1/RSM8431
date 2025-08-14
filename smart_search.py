from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re
import math

#error handling
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
    """Lowercase, strip, and remove non-alphanum (keeps spaces and hyphens)."""
    try:
        t = "" if t is None else str(t)
        t = t.lower().strip()
        t = re.sub(r"[^a-z0-9\s\-]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t
    except Exception:
        # Never raise from a normalizer; return a safe string.
        return ""


def tokenize(s: str) -> List[str]:
    """Tokenize a string into whitespace-separated words after normalization."""
    s = normalize_token(s)
    return [w for w in s.split(" ") if w]


def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity on token sets; safe on empties."""
    try:
        A, B = set(a or []), set(b or [])
        if not A and not B:
            return 0.0
        return len(A & B) / float(len(A | B))
    except Exception:
        return 0.0


def alias_expand(tokens: List[str]) -> List[str]:
    """Expand tokens to include canonical tags and aliases, idempotently."""
    try:
        expanded = set(tokens or [])
        for canon, aliases in CANONICAL_TAGS.items():
            if canon in expanded or any(a in expanded for a in aliases):
                expanded.add(canon)
                expanded |= set(aliases)
        return list(expanded)
    except Exception:
        # Fall back to the original list if anything goes wrong
        return list(tokens or [])


class SmartSearch:
    def __init__(self, properties_df: "pd.DataFrame"):
        """
        Build a lightweight searchable corpus from key CSV columns.
        Never raises on dirty/missing data; falls back to safe defaults.
        """
        # Defensive copy+coercion
        try:
            df = properties_df.copy()
        except Exception:
            # If a non-DataFrame sneaks in, coerce it into one
            try:
                df = pd.DataFrame(properties_df)
            except Exception:
                df = pd.DataFrame()

        # Normalize columns to lowercase strings
        try:
            cols = [str(c).lower() for c in df.columns]
            df.columns = cols
        except Exception:
            # In extreme cases, rebuild with stringified data
            df = pd.DataFrame(df)
            df.columns = [str(c).lower() for c in df.columns]

        # Collect candidate text parts (or fall back to all columns)
        parts = []
        for col in ["location", "type", "features", "tags", "description"]:
            if col in df.columns:
                try:
                    parts.append(df[col].fillna("").astype(str))
                except Exception:
                    # Be resilient to weird dtypes
                    parts.append(df[col].map(lambda x: "" if x is None else str(x)))

        if parts:
            try:
                full = parts[0]
                for p in parts[1:]:
                    full = full + " | " + p
            except Exception:
                # Fallback: concatenate row-wise across whatever we have
                full = pd.Series([""] * len(df))
                for p in parts:
                    try:
                        full = full + " | " + p.astype(str)
                    except Exception:
                        pass
        else:
            # Fallback to all columns if expected ones are missing
            try:
                full = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1)
            except Exception:
                # Last resort: empty strings
                full = pd.Series([""] * len(df))

        try:
            df["__fulltext__"] = full.apply(normalize_token)
        except Exception:
            # Ensure the column exists
            try:
                df["__fulltext__"] = pd.Series([normalize_token(x) for x in full])
            except Exception:
                df["__fulltext__"] = ""

        self.df = df

    def find_candidates(self, query: str, top_k: int = 50) -> "pd.DataFrame":
        """
        Rank rows by Jaccard similarity between query tokens and row tokens.
        Returns a DataFrame with a 'semantic_score' column. Safe on bad input.
        """
        # Normalize inputs
        try:
            q = "" if query is None else str(query)
        except Exception:
            q = ""
        q_tokens = alias_expand(tokenize(q))

        # Bound and sanitize top_k
        try:
            if not isinstance(top_k, int):
                top_k = int(top_k)
        except Exception:
            top_k = 50
        top_k = max(1, min(top_k, max(1, len(self.df))))

        # Ensure __fulltext__ exists (self-heal if needed)
        if "__fulltext__" not in self.df.columns:
            try:
                self.df["__fulltext__"] = ""
            except Exception:
                pass

        texts = []
        try:
            # Make sure we always iterate strings
            texts = [normalize_token(t) for t in self.df["__fulltext__"].fillna("").astype(str).tolist()]
        except Exception:
            # Very defensive fallback
            try:
                texts = [normalize_token(t) for t in list(self.df.get("__fulltext__", []))]
            except Exception:
                texts = []

        scores = []
        try:
            for text in texts:
                t_tokens = alias_expand(tokenize(text))
                s = jaccard(q_tokens, t_tokens)
                scores.append(s)
        except Exception:
            # If something unexpected happens, align lengths safely
            scores = [0.0] * len(texts)

        try:
            scored = self.df.copy()
        except Exception:
            scored = pd.DataFrame(self.df)

        try:
            scored["semantic_score"] = scores + [0.0] * max(0, len(scored) - len(scores))
        except Exception:
            # If assignment fails, ensure the column exists
            scored["semantic_score"] = 0.0

        try:
            scored = scored.sort_values("semantic_score", ascending=False).head(top_k)
        except Exception:
            # If sort fails, just take head
            scored = scored.head(top_k)

        return scored

    def canonicalize_tags(self, raw_tags: List[str]) -> List[str]:
        """Return a sorted list including canonical tags and their aliases."""
        try:
            out = set()
            for t in (raw_tags or []):
                t = normalize_token(t)
                if not t:
                    continue
                out.add(t)
                for canon, aliases in CANONICAL_TAGS.items():
                    if t == canon or t in aliases:
                        out.add(canon)
                        out |= aliases
            return sorted(out)
        except Exception:
            # On any failure, return a safe, normalized copy
            try:
                return sorted(set(normalize_token(t) for t in (raw_tags or []) if normalize_token(t)))
            except Exception:
                return []
