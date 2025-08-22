"""SmartSearch: keyword matching over tabular text.
- Normalization and tokenization of free text
- Query expansion via a simple alias dictionary (e.g., "hottub" → "hot tub")
- Row-level scoring using Jaccard similarity on unique tokens
- A pandas-friendly class (`SmartSearch`) that ranks records by match quality

Intended for small/medium datasets where a fast, dependency-light search is useful.
"""

from typing import List
import pandas as pd
import re

# Canonical tags and their common aliases. If a token matches either a canon or an alias,
# the whole set (canon + aliases) is injected to improve recall.
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
    """Normalize a string into a search-friendly lowercase token line.

    Steps:
    1) Lowercase and coerce to string; map None → "".
    2) Replace non [a-z0-9 -] chars with spaces.
    3) Collapse multiple spaces and trim.

    Args:
        s: Raw string or value convertible to string.

    Returns:
        A normalized, space-delimited string suitable for tokenization.
    """
    s = "" if s is None else str(s).lower()
    # Keep alphanumerics, spaces, and dashes; strip everything else.
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    # Squash runs of whitespace to a single space.
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    """Split a string into simple tokens after normalization.

    Args:
        s: Input string.

    Returns:
        List of non-empty tokens.
    """
    # Normalize first to ensure consistent splitting and matching.
    return [t for t in normalize_token(s).split(" ") if t]

def jaccard(a: List[str], b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists.

    Jaccard(A,B) = |A ∩ B| / |A ∪ B|. Empty vs empty returns 0.0.

    Args:
        a: First list of tokens.
        b: Second list of tokens.

    Returns:
        Similarity in [0.0, 1.0]. Returns 0.0 on exceptions.
    """
    try:
        A, B = set(a or []), set(b or [])
        if not A and not B:
            return 0.0
        return len(A & B) / float(len(A | B))
    except Exception:
        return 0.0

def alias_expand(tokens: List[str]) -> List[str]:
    """Expand a token list with canonical tags and aliases.

    If any token matches a canonical tag or any of its aliases,
    the result includes both the canonical form and all aliases.

    Args:
        tokens: Input tokens.

    Returns:
        Expanded token list (order not guaranteed).
    """
    try:
        expanded = set(tokens or [])
        # Inject canon + aliases whenever any one is present.
        for canon, aliases in CANONICAL_TAGS.items():
            if canon in expanded or any(a in expanded for a in aliases):
                expanded.add(canon)
                expanded |= set(aliases)
        return list(expanded)
    except Exception:
        # Preserve original tokens if expansion fails.
        return list(tokens or [])


class SmartSearch:
    """Search helper that ranks DataFrame rows by Jaccard over aggregated text.

    It builds a hidden '__fulltext__' column by concatenating commonly useful
    descriptive fields, normalizes them, and then scores rows against a query.

    Typical columns considered (if present):
        'location', 'property_type', 'features', 'tags', 'environment', 'description'
    """

    def __init__(self, properties_df: "pd.DataFrame"):
        """Create a SmartSearch instance from a properties DataFrame.

        Attempts to be resilient to non-DataFrame inputs by copying or
        converting where possible, and to missing columns by falling back
        to a whole-row string join.

        Args:
            properties_df: A pandas DataFrame (or convertible) containing
                           property metadata to search.
        """
        try:
            df = properties_df.copy()
        except Exception:
            try:
                df = pd.DataFrame(properties_df)
            except Exception:
                df = pd.DataFrame()

        # Normalize header names to lowercase for predictable access.
        try:
            df.columns = [str(c).lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame(df)
            df.columns = [str(c).lower() for c in df.columns]

        # Build an aggregated text field from common descriptive columns.
        parts = []
        for col in ["location", "property_type", "features", "tags", "environment", "description"]:
            if col in df.columns:
                try:
                    parts.append(df[col].fillna("").astype(str))
                except Exception:
                    # Fallback: map None to "" and coerce to str.
                    parts.append(df[col].map(lambda x: "" if x is None else str(x)))

        # Concatenate parts with a visible delimiter; fallback to whole row.
        if parts:
            try:
                full = parts[0]
                for p in parts[1:]:
                    full = full + " | " + p
            except Exception:
                # Build an empty series, then add piecewise.
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

        # Normalize the aggregated text into a hidden searchable column.
        try:
            df["__fulltext__"] = full.apply(normalize_token)
        except Exception:
            try:
                df["__fulltext__"] = pd.Series([normalize_token(x) for x in full])
            except Exception:
                df["__fulltext__"] = ""

        self.df = df

    def find_candidates(self, query: str, top_k: int = 300) -> pd.DataFrame:
        """Return top-k rows ranked by Jaccard similarity to a query.

        Steps:
            1) Tokenize and alias-expand the query.
            2) Tokenize each row's '__fulltext__'.
            3) Score with Jaccard and sort descending.

        Args:
            query: Free-text query string.
            top_k: Number of rows to return (default 300).

        Returns:
            A new DataFrame, sorted by 'semantic_score' (descending),
            containing the original columns plus 'semantic_score'.
        """
        # Prepare the query tokens with alias expansion to improve recall.
        q = alias_expand(tokenize(query))

        # Row scoring function—kept small for map() usage.
        def score_row(text):
            return jaccard(set(q), set(tokenize(text)))

        # Compute scores.
        try:
            scores = self.df["__fulltext__"].map(score_row)
        except Exception:
            scores = pd.Series([0.0] * len(self.df))

        # Return a copy with the score column, sorted best-first.
        out = self.df.copy()
        out["semantic_score"] = scores
        return out.sort_values("semantic_score", ascending=False).head(top_k)
