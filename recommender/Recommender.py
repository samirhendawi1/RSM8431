from typing import Optional, List, Dict, Any
import math
import pandas as pd
import re

# Ensure pandas is imported for DataFrame handling
def _tok(s: Any):
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s) #replace any ^a-z0-9\s\- with space
    return [w for w in re.sub(r"\s+", " ", s).split(" ") if w]

#jecard similarity between two sets(user preferences and property attributes)
def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / float(len(a | b)) if (a or b) else 0.0

class Recommender:
    """
    Final score is a dynamic, data-driven blend of:
      - env_score        (environment/type alignment)
      - budget_score     (price closeness to user's range)
      - group_score      (capacity fit)
      - tag_feature_score (LLM tags/features vs row tokens)
      - location_score   (LLM locations vs row location)
      - llm_id_boost     (direct id match boost)
      - llm_similarity   (broad overlap with LLM tags/features/env/type)

    Weights auto-adjust based on which signals are present.
    """
    def __init__(self, top_k: int = 5, base_weights: Dict[str, float] = None):
        self.top_k = int(top_k)
        self.base_weights = base_weights or {
            "env": 0.22,
            "budget": 0.22,
            "group": 0.18,
            "tag_feature": 0.18,
            "location": 0.10,
            "llm_sim": 0.10,
        }
        # Note: llm_id_boost is additive, not part of the normalized blend
        self.id_boost_weight = 0.20  # 0/1 flag → +0.20


    def _location_score(self, row_loc: str, wanted_locs: List[str]) -> float:
        if not wanted_locs: # no location preference, return location score 0
            return 0.0
        rl = (row_loc or "").lower()
        if not rl: # if there if no location in the row, return location score 0
            return 0.0
        for w in wanted_locs:
            if not w:
                continue
            w = str(w).lower()
            if w in rl: # for exact token match, return location score 1
                return 1.0
        # soft token overlap
        return _jaccard(set(_tok(rl)), set(_tok(" ".join(map(str, wanted_locs)))))

    # Scores for each row based on user preferences and LLM
    def _env_score(self, row: pd.Series, user_env: str, llm_envs: List[str]) -> float:
        tokens = set(_tok(row.get("environment","")) + _tok(row.get("tags","")) + _tok(row.get("property_type",""))) #tokens from properties list environment, tags and property type
        wants = set(_tok(user_env)) | set(_tok(" ".join(llm_envs or []))) #tokens from user input union with LLM environments
        if not wants: #if no environment preference, set as neutral score 0.5
            return 0.5
        return _jaccard(tokens, wants)

    # Budget score based on price vs user budget range
    def _budget_score(self, price: float, bmin: float, bmax: float) -> float:
        try:
            price = float(price or 0)
        except Exception:
            price = 0.0 #fall back to 0 if conversion fails
        if (not bmin and not bmax) or (float(bmin)==0 and float(bmax)==0):
            return 0.5 #if there is no budget preference or the max/min budget are both 0, set as neutral score 0.5
        lo = min(float(bmin or 0), float(bmax or 0))
        hi = max(float(bmin or 0), float(bmax or 0))
        rng = max(hi - lo, 1e-9)
        mid = (lo + hi)/2.0
        return max(0.0, 1.0 - abs(price - mid)/rng)

    # Group size score based on min/max guests
    def _group_score(self, row: pd.Series, gsize: int) -> float:
        try:
            lo = int(row.get("min_guests", 0) or 0)
            hi = int(row.get("max_guests", 0) or 0)
        except Exception:
            lo, hi = 0, 0
        if not gsize or gsize <= 0:
            return 0.5
        if lo <= gsize <= hi:
            return 1.0
        dist = (lo - gsize) if gsize < lo else (gsize - hi)
        return math.exp(-0.5 * max(0, dist))

    # Tag/feature score based on LLM hints vs row tokens
    def _tag_feature_score(self, row: pd.Series, llm_tags: List[str], llm_features: List[str]) -> float:
        wanted = set(_tok(" ".join((llm_tags or []) + (llm_features or []))))
        if not wanted:
            return 0.0
        row_tokens = set(_tok(" ".join([
            str(row.get("__fulltext__", "")),
            str(row.get("tags","")),
            str(row.get("features","")),
            str(row.get("environment","")),
            str(row.get("property_type","")),
        ])))
        return _jaccard(row_tokens, wanted)

    # board similarity score based on LLM
    def _llm_similarity(self, row: pd.Series, llm_tags: List[str], llm_features: List[str], llm_envs: List[str]) -> float:
        pool = set(_tok(" ".join((llm_tags or []) + (llm_features or []) + (llm_envs or []))))
        if not pool:
            return 0.0
        row_tokens = set(_tok(" ".join([
            str(row.get("__fulltext__", "")),
            str(row.get("tags","")),
            str(row.get("features","")),
            str(row.get("environment","")),
            str(row.get("property_type","")),
        ])))
        return _jaccard(row_tokens, pool)

    #Increse weights for signals that are present
    def _dynamic_weights(self, user_env: str, bmin: float, bmax: float, gsize: int,
                         has_tags: bool, has_features: bool, has_locs: bool, has_envs: bool) -> Dict[str, float]:
        # Presence checks
        budget_present = bool((bmin or bmax) and not (float(bmin) == 0 and float(bmax) == 0))
        group_present  = bool(gsize and gsize > 0)
        env_present    = bool((str(user_env).strip() if user_env else "") or has_envs)
        tag_present    = bool(has_tags or has_features)
        loc_present    = bool(has_locs)
        sim_present    = bool(has_envs or has_tags or has_features)  # llm_sim only meaningful if any LLM hint exists

        # Collect present signals with base magnitudes
        present = {}
        if env_present:     present["env"]         = self.base_weights.get("env", 0.0)
        if budget_present:  present["budget"]      = self.base_weights.get("budget", 0.0)
        if group_present:   present["group"]       = self.base_weights.get("group", 0.0)
        if tag_present:     present["tag_feature"] = self.base_weights.get("tag_feature", 0.0)
        if loc_present:     present["location"]    = self.base_weights.get("location", 0.0)
        if sim_present:     present["llm_sim"]     = self.base_weights.get("llm_sim", 0.0)

        # Normalize only the present ones
        if present:
            total = sum(present.values())
            if total > 0:
                present = {k: v / total for k, v in present.items()}
            else:
                # Degenerate case: if all base magnitudes were zero, split equally among present
                eq = 1.0 / len(present)
                present = {k: eq for k in present.keys()}
        else:
            # No signals present: (choose one)
            # Option A (robust): equal weights across all signals
            eq = 1.0 / 6.0
            present = {k: eq for k in ["env","budget","group","tag_feature","location","llm_sim"]}
            # Option B (strict): all zero weights — uncomment next line and comment Option A if you prefer hard-zero fallback
            # present = {k: 0.0 for k in ["env","budget","group","tag_feature","location","llm_sim"]}

        # Return full dict with zeros for any missing keys
        full = {k: 0.0 for k in ["env","budget","group","tag_feature","location","llm_sim"]}
        full.update(present)
        return full


    def recommend(
            self,
            user: Any,
            df: pd.DataFrame,
            *,
            llm_hints: Optional[Dict[str, List[str]]] = None,
            llm_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        #guard against empty dataframe
        if df is None or len(df) == 0:
            return pd.DataFrame()
        d = df.copy()

        # Unpack user prefs
        user_env = getattr(user, "environment", "") or ""
        bmin = float(getattr(user, "budget_min", 0) or 0)
        bmax = float(getattr(user, "budget_max", 0) or 0)
        gsize = int(getattr(user, "group_size", 0) or 0)

        # Unpack LLM hints if any, if not present use empty lists
        hints = llm_hints or {}
        tags = [t for t in (hints.get("tags") or []) if t]
        feats = [t for t in (hints.get("features") or []) if t]
        locs = [t for t in (hints.get("locations") or []) if t]
        envs = [t for t in (hints.get("environments") or []) if t]

        # Scores, calling the scoring function from above
        d["env_score"] = d.apply(lambda r: self._env_score(r, user_env, envs), axis=1)
        d["budget_score"] = d["nightly_price"].apply(lambda p: self._budget_score(p, bmin, bmax))
        d["group_score"] = d.apply(lambda r: self._group_score(r, gsize), axis=1)
        d["tag_feature_score"] = d.apply(lambda r: self._tag_feature_score(r, tags, feats), axis=1)
        d["location_score"] = d["location"].apply(lambda loc: self._location_score(loc, locs))
        d["llm_similarity_score"] = d.apply(lambda r: self._llm_similarity(r, tags, feats, envs), axis=1)

        # ID boost (string compare, robust to "P-101" etc.)
        if llm_ids and "property_id" in d.columns:
            idset = set(str(x).strip() for x in llm_ids if x is not None)
            d["llm_id_hit"] = d["property_id"].astype(str).str.strip().isin(idset).astype(float)
        else:
            d["llm_id_hit"] = 0.0

        # Dynamic weights
        W = self._dynamic_weights(
            user_env=user_env, bmin=bmin, bmax=bmax, gsize=gsize,
            has_tags=bool(tags), has_features=bool(feats),
            has_locs=bool(locs), has_envs=bool(envs)
        )

        # Final blended score + additive ID boost
        d["fit_score"] = (
                W["env"]        * d["env_score"] +
                W["budget"]     * d["budget_score"] +
                W["group"]      * d["group_score"] +
                W["tag_feature"]* d["tag_feature_score"] +
                W["location"]   * d["location_score"] +
                W["llm_sim"]    * d["llm_similarity_score"] +
                self.id_boost_weight * d["llm_id_hit"]
        )

        cols = [
            "property_id","location","environment","property_type","nightly_price",
            "min_guests","max_guests","features","tags",
            # diagnostics (keep if you want to inspect tuning)
            # "env_score","budget_score","group_score",
            # "tag_feature_score","location_score","llm_similarity_score","llm_id_hit",
            "fit_score"
        ]
        d = d.sort_values("fit_score", ascending=False).reset_index(drop=True)
        return d[[c for c in cols if c in d.columns]].head(self.top_k)
