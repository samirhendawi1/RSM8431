from typing import Optional, List
import math
import pandas as pd
import re

def _tok(s: str):
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return [w for w in re.sub(r"\s+", " ", s).split(" ") if w]

class Recommender:
    """
    Composite scoring with tunable weights and explicit LLM influence.

    final fit_score =
        we * env_score
      + wb * budget_score
      + wg * group_score
      + wllm * llm_similarity_score
      + llm_boost   # (tag/id hits as small additive bump)

    - llm_similarity_score: token overlap between LLM tags and each row's text.
    - llm_boost: small additive bump for explicit tag/id hits.
    """

    def __init__(self, top_k: int = 5, weights=(0.4, 0.4, 0.2), llm_weight: float = 0.25):
        self.top_k = int(top_k)
        self.weights = weights  # (env, budget, group)
        self.llm_weight = float(llm_weight)

    def recommend(
            self,
            user,
            df: pd.DataFrame,
            llm_tags: Optional[List[str]] = None,
            llm_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        d = df.copy()

        # ----- ENV -----
        env_tokens = set(_tok(getattr(user, "environment", "")))
        def env_score(row):
            tokens = set(_tok(row.get("environment","")) + _tok(row.get("tags","")) + _tok(row.get("property_type","")))
            if not env_tokens: return 0.5
            inter = len(tokens & env_tokens); union = len(tokens | env_tokens)
            return (inter/union) if union else 0.0
        d["env_score"] = d.apply(env_score, axis=1)

        # ----- BUDGET -----
        bmin = float(getattr(user,"budget_min",0) or 0)
        bmax = float(getattr(user,"budget_max",0) or 0)
        if bmax < bmin: bmin, bmax = bmax, bmin
        rng = max(bmax - bmin, 1e-9)
        avg = (bmin + bmax)/2.0
        d["nightly_price"] = pd.to_numeric(d["nightly_price"], errors="coerce").fillna(0)
        if bmin==0 and bmax==0:
            d["budget_score"] = 0.5  # neutral if no budget provided
        else:
            d["budget_score"] = (1.0 - (d["nightly_price"] - avg).abs() / rng).clip(0,1)

        # ----- GROUP -----
        gsize = int(getattr(user,"group_size",0) or 0)
        d["min_guests"] = pd.to_numeric(d["min_guests"], errors="coerce").fillna(0).astype(int)
        d["max_guests"] = pd.to_numeric(d["max_guests"], errors="coerce").fillna(0).astype(int)
        def group_score(row):
            lo, hi = row["min_guests"], row["max_guests"]
            if gsize<=0: return 0.5
            if lo <= gsize <= hi: return 1.0
            dist = (lo - gsize) if gsize < lo else (gsize - hi)
            return math.exp(-0.5 * max(0, dist))
        d["group_score"] = d.apply(group_score, axis=1)

        # ----- LLM SIMILARITY SCORE (new) -----
        tags_lower = [t.lower() for t in (llm_tags or []) if isinstance(t, str) and t.strip()]
        tag_set = set(tags_lower)
        def llm_sim(row):
            if not tag_set:
                return 0.0
            text = " ".join([
                str(row.get("__fulltext__", "")),
                str(row.get("tags","")),
                str(row.get("features","")),
                str(row.get("environment","")),
                str(row.get("property_type",""))
            ]).lower()
            text_tokens = set(_tok(text))
            if not text_tokens:
                return 0.0
            inter = len(text_tokens & tag_set); union = len(text_tokens | tag_set)
            return (inter/union) if union else 0.0
        d["llm_similarity_score"] = d.apply(llm_sim, axis=1)

        # ----- LLM BOOST (existing, kept) -----
        if tags_lower:
            d["llm_tag_hit"] = d.get("__fulltext__", (d.get("tags","")+" "+d.get("features",""))).astype(str).str.lower().apply(
                lambda t: any(tag in t for tag in tags_lower)
            ).astype(float)
        else:
            d["llm_tag_hit"] = 0.0
        if llm_ids and "property_id" in d.columns:
            d["llm_id_hit"] = d["property_id"].isin(llm_ids).astype(float)
        else:
            d["llm_id_hit"] = 0.0
        d["llm_boost"] = 0.15 * d["llm_tag_hit"] + 0.20 * d["llm_id_hit"]

        # ----- Final -----
        we, wb, wg = self.weights
        total = max(we+wb+wg, 1e-9)
        we, wb, wg = we/total, wb/total, wg/total

        d["fit_score"] = (
                we * d["env_score"] +
                wb * d["budget_score"] +
                wg * d["group_score"] +
                self.llm_weight * d["llm_similarity_score"] +
                d["llm_boost"]
        )

        cols = [
            "property_id","location","environment","property_type","nightly_price",
            "min_guests","max_guests","features","tags",
            "env_score","budget_score","group_score",
            "llm_similarity_score","llm_boost","fit_score"
        ]
        d = d.sort_values("fit_score", ascending=False).reset_index(drop=True)
        return d[[c for c in cols if c in d.columns]].head(self.top_k)
