## im making an edit to this.

from typing import List
import pandas as pd
import re

CANONICAL_TAGS = {
    "hot tub": ["hottub", "jacuzzi"],
    "pool": ["swimming pool"],
    "wifi": ["wi-fi", "internet"],
    "beach": ["oceanfront","seaside","coastal"],
    "city": ["downtown","urban"],
    "mountain": ["alpine"],
    "lake": ["lakeside"],
}

def normalize_token(s: str) -> str:
    s = "" if s is None else str(s).lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [t for t in normalize_token(s).split(" ") if t]

def jaccard(a: List[str], b: List[str]) -> float:
    try:
        A, B = set(a or []), set(b or [])
        if not A and not B:
            return 0.0
        return len(A & B) / float(len(A | B))
    except Exception:
        return 0.0

def alias_expand(tokens: List[str]) -> List[str]:
    try:
        expanded = set(tokens or [])
        for canon, aliases in CANONICAL_TAGS.items():
            if canon in expanded or any(a in expanded for a in aliases):
                expanded.add(canon)
                expanded |= set(aliases)
        return list(expanded)
    except Exception:
        return list(tokens or [])


class SmartSearch:
    def __init__(self, properties_df: "pd.DataFrame"):
        try:
            df = properties_df.copy()
        except Exception:
            try:
                df = pd.DataFrame(properties_df)
            except Exception:
                df = pd.DataFrame()

        # Lowercase headers
        try:
            df.columns = [str(c).lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame(df)
            df.columns = [str(c).lower() for c in df.columns]

        # Build a text field from typical columns (include property_type)
        parts = []
        for col in ["location", "property_type", "features", "tags", "environment", "description"]:
            if col in df.columns:
                try:
                    parts.append(df[col].fillna("").astype(str))
                except Exception:
                    parts.append(df[col].map(lambda x: "" if x is None else str(x)))

        if parts:
            try:
                full = parts[0]
                for p in parts[1:]:
                    full = full + " | " + p
            except Exception:
                full = pd.Series([""] * len(df))
                for p in parts:
                    try:
                        full = full + " | " + p.astype(str)
                    except Exception:
                        pass
        else:
            try:
                full = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1)
            except Exception:
                full = pd.Series([""] * len(df))

        try:
            df["__fulltext__"] = full.apply(normalize_token)
        except Exception:
            try:
                df["__fulltext__"] = pd.Series([normalize_token(x) for x in full])
            except Exception:
                df["__fulltext__"] = ""

        self.df = df

    def find_candidates(self, query: str, top_k: int = 300) -> pd.DataFrame:
        q = alias_expand(tokenize(query))
        def score_row(text):
            return jaccard(set(q), set(tokenize(text)))
        try:
            scores = self.df["__fulltext__"].map(score_row)
        except Exception:
            scores = pd.Series([0.0] * len(self.df))
        out = self.df.copy()
        out["semantic_score"] = scores
        return out.sort_values("semantic_score", ascending=False).head(top_k)
