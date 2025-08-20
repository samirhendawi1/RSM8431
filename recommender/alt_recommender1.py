from typing import Optional, List
import math
import pandas as pd
import re

def _tok(s: str):
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [w for w in s.split(" ") if w]

class Recommender:
    """
    final fit_score =
        (we*env + wb*budget + wg*group)   # weights renormalized over ACTIVE base signals only
        + wllm*llm_similarity
        + wloc*location_score
        + llm_boost
    Notes:
      - wloc contributes only if user provided desired locations.
      - Existing API is unchanged.
    """
    def __init__(self, top_k: int = 5, weights=(0.4, 0.4, 0.2), llm_weight: float = 0.25, loc_weight: float = 0.15):
        self.top_k = int(top_k)
        self.weights = tuple(float(x) for x in weights)
        self.llm_weight = float(llm_weight)
        self.loc_weight = float(loc_weight) 

    # -------- helpers --------
    def _normalize_locations(self, user) -> List[str]:
        """
        Extract desired locations from common user fields without changing the public API.
        Accepts either a string or a list/tuple/set of strings.
        """
        wanted = []
        # check a few likely attribute names; harmless if absent
        for attr in ("desired_locations", "locations", "location_preference", "location", "destination", "city", "region"):
            val = getattr(user, attr, None)
            if val is None:
                continue
            if isinstance(val, str):
                if val.strip():
                    wanted.append(val.strip())
            elif isinstance(val, (list, tuple, set)):
                for x in val:
                    if isinstance(x, str) and x.strip():
                        wanted.append(x.strip())
        # de-dup & lowercase comparisons will be done in scorer
        return wanted

    def _location_score(self, row_loc: str, wanted_locs: List[str]) -> float:
        """
        Hard 1.0 if any desired location is a substring of the row location (case-insensitive).
        Else, soft token overlap (Jaccard) as a fallback.
        """
        if not wanted_locs:
            return 0.0
        rl = (row_loc or "").lower()
        if not rl:
            return 0.0

        # hard hit if any desired string appears as substring
        for w in wanted_locs:
            w = (w or "").strip().lower()
            if w and w in rl:
                return 1.0

        # soft overlap fallback
        row_tokens = set(_tok(rl))
        want_tokens = set(_tok(" ".join(wanted_locs)))
        if not row_tokens or not want_tokens:
            return 0.0
        inter = len(row_tokens & want_tokens)
        union = len(row_tokens | want_tokens)
        return (inter / union) if union else 0.0

    # -------- main --------
    def recommend(self, user, df: pd.DataFrame,
                  llm_tags: Optional[List[str]] = None,
                  llm_ids: Optional[List] = None) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()

        d = df.copy()

        # Ensure expected text/num columns exist (avoid KeyErrors)
        for col in ["location", "environment", "tags", "property_type", "features", "__fulltext__"]:
            if col not in d.columns:
                d[col] = ""
        d["nightly_price"] = pd.to_numeric(d.get("nightly_price", 0), errors="coerce").fillna(0.0)
        d["min_guests"]   = pd.to_numeric(d.get("min_guests", 0), errors="coerce").fillna(0).astype(int)
        d["max_guests"]   = pd.to_numeric(d.get("max_guests", 0), errors="coerce").fillna(0).astype(int)

        # ----------------------------
        # ENVIRONMENT (Dice similarity; active only if user provided env)
        # ----------------------------
        env_tokens = set(_tok(getattr(user, "environment", "")))
        def env_score(row):
            if not env_tokens:
                return 0.5  # neutral placeholder; ignored if env inactive
            tokens = set(_tok(row.get("environment","")) +
                         _tok(row.get("tags","")) +
                         _tok(row.get("property_type","")))
            if not tokens:
                return 0.0
            inter = len(tokens & env_tokens)
            denom = len(tokens) + len(env_tokens)
            return (2.0 * inter / denom) if denom else 0.0
        d["env_score"] = d.apply(env_score, axis=1)
        env_active = bool(env_tokens)

        # ----------------------------
        # BUDGET (Gaussian falloff outside the band; inside = 1.0)
        # ----------------------------
        bmin = float(getattr(user, "budget_min", 0) or 0.0)
        bmax = float(getattr(user, "budget_max", 0) or 0.0)
        if bmax < bmin:
            bmin, bmax = bmax, bmin

        def _budget_score(price: float) -> float:
            # No budget → neutral (ignored if inactive)
            if bmin == 0.0 and bmax == 0.0:
                return 0.5

            # Single-point target → Gaussian around the target
            if bmax == bmin:
                center = bmin
                sigma = max(0.15 * max(center, 1.0), 1.0)  # 15% of target or ≥1
                z = (price - center) / sigma
                return math.exp(-0.5 * z * z)

            # Range: inside gets full score; outside decays as Gaussian from nearest edge
            if bmin <= price <= bmax:
                return 1.0
            dist = (bmin - price) if price < bmin else (price - bmax)
            sigma = max(0.35 * (bmax - bmin), 1.0)        # 35% of range or ≥1
            z = dist / sigma
            return math.exp(-0.5 * z * z)

        d["budget_score"] = d["nightly_price"].apply(_budget_score)
        budget_active = not (bmin == 0.0 and bmax == 0.0)

        # ----------------------------
        # GROUP SIZE (softer exponential falloff outside [min,max])
        # ----------------------------
        gsize = int(getattr(user, "group_size", 0) or 0)
        def group_score(row):
            lo, hi = row["min_guests"], row["max_guests"]
            if gsize <= 0:
                return 0.5  # neutral; ignored if inactive
            if lo == 0 and hi == 0:
                return 0.5  # unknown capacity
            if lo <= gsize <= (hi if hi > 0 else gsize):
                return 1.0
            dist = (gsize - hi) if (hi > 0 and gsize > hi) else max(0, lo - gsize)
            tau = 1.5
            return math.exp(-dist / tau)

        d["group_score"] = d.apply(group_score, axis=1)
        group_active = bool(gsize > 0)

        # ----------------------------
        # LOCATION (active only if user provided desired locations)
        # ----------------------------
        wanted_locs = self._normalize_locations(user)
        d["location_score"] = d["location"].apply(lambda loc: self._location_score(loc, wanted_locs))
        loc_active = bool(wanted_locs)

        # ----------------------------
        # LLM SIMILARITY (recall over user-requested tags)
        # ----------------------------
        tags_lower = [t.strip().lower() for t in (llm_tags or []) if isinstance(t, str) and t.strip()]
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
            inter = len(text_tokens & tag_set)
            return inter / max(len(tag_set), 1)  # coverage of requested tags (0..1)
        d["llm_similarity_score"] = d.apply(llm_sim, axis=1)

        # ----------------------------
        # LLM BOOSTS
        # ----------------------------
        if tags_lower:
            blob = (d.get("__fulltext__", "") + " " +
                    d.get("tags", "") + " " +
                    d.get("features", "")).astype(str).str.lower()
            d["llm_tag_hit"] = blob.apply(lambda t: float(any(tag in t for tag in tags_lower)))
        else:
            d["llm_tag_hit"] = 0.0

        if llm_ids and "property_id" in d.columns:
            ids_as_str = set(str(x).strip() for x in llm_ids if x is not None)
            d["llm_id_hit"] = d["property_id"].astype(str).str.strip().isin(ids_as_str).astype(float)
        else:
            d["llm_id_hit"] = 0.0

        d["llm_boost"] = 0.10 * d["llm_tag_hit"] + 0.15 * d["llm_id_hit"]

        # ----------------------------
        # FINAL SCORE
        # ----------------------------
        we, wb, wg = self.weights
        # base weights only across ACTIVE base signals
        base_weights = [
            we if env_active else 0.0,
            wb if budget_active else 0.0,
            wg if group_active else 0.0,
        ]
        total_base = sum(base_weights)

        if total_base > 0:
            we_eff, wb_eff, wg_eff = (base_weights[0]/total_base,
                                      base_weights[1]/total_base,
                                      base_weights[2]/total_base)
            base_score = (we_eff * d["env_score"]
                          + wb_eff * d["budget_score"]
                          + wg_eff * d["group_score"])
        else:
            base_score = pd.Series(0.5, index=d.index)  # no active base constraints → neutral base

        llm_w_eff = self.llm_weight if tag_set else 0.0
        loc_w_eff = self.loc_weight if loc_active else 0.0

        d["fit_score"] = (base_score
                          + llm_w_eff * d["llm_similarity_score"]
                          + loc_w_eff * d["location_score"]
                          + d["llm_boost"])

        cols = ["property_id","location","environment","property_type","nightly_price",
                "min_guests","max_guests","features","tags","fit_score"]
        d = d.sort_values("fit_score", ascending=False).reset_index(drop=True)
        return d[[c for c in cols if c in d.columns]].head(self.top_k)
