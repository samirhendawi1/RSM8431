import math
from typing import Optional, List, Tuple, Union, Iterable, Set

import numpy as np
import pandas as pd


def _as_lower_str(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).lower()


def _tokenize(x: str) -> List[str]:
    x = _as_lower_str(x)
    # Split on non-alphanumerics; keep simple words
    tokens = []
    cur = []
    for ch in x:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return [t for t in tokens if t]


def _to_set(iterable: Iterable[str]) -> Set[str]:
    return set([t for t in iterable if t])


class Recommender:
    """
    Improved recommender with soft tag matching and robust budget scoring.

    Backward-compat behavior:
      - recommend(user, df)                      -> works (default equal weights)
      - recommend(user, df, 1|2|3)               -> importance choice (1=env, 2=price, 3=equal)
      - recommend(user, df, "wifi pool")         -> treated as keyword BOOST (not a strict filter)
      - recommend(user, df, (w_env, w_budget))   -> explicit weights tuple
      - recommend(user, df, importance=..., query=..., weight_env=..., weight_budget=...)
    """

    def __init__(self, top_k: int = 5, echo_table: bool = True):
        self.top_k = top_k
        self.echo_table = echo_table

    # NOTE: flexible signature to avoid "takes 3 positional args but 4 were given"
    def recommend(
            self,
            user,
            properties_df: pd.DataFrame,
            extra: Optional[Union[int, float, str, Tuple[float, float], List[float]]] = None,
            **kwargs,
    ) -> pd.DataFrame:
        if properties_df is None or len(properties_df) == 0:
            return pd.DataFrame()

        df = properties_df.copy()

        # Normalize column names: allow both "type" and "ptype"
        if "type" not in df.columns and "ptype" in df.columns:
            df["type"] = df["ptype"]

        # Basic required columns
        required = ["nightly_price"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Recommender: missing required column(s): {missing}")

        # ---------------- Parse arguments / weights ----------------
        importance = kwargs.pop("importance", None)
        query = kwargs.pop("query", None)
        weight_env = kwargs.pop("weight_env", None)
        weight_budget = kwargs.pop("weight_budget", None)

        if extra is not None:
            if isinstance(extra, (tuple, list)) and len(extra) >= 2:
                weight_env = float(extra[0])
                weight_budget = float(extra[1])
            elif isinstance(extra, (int, float)):
                if int(extra) in (1, 2, 3):
                    importance = int(extra)
            elif isinstance(extra, str):
                if extra.strip().isdigit():
                    val = int(extra.strip())
                    if val in (1, 2, 3):
                        importance = val
                else:
                    query = extra

        if weight_env is None or weight_budget is None:
            if importance == 1:
                weight_env, weight_budget = 0.7, 0.3
            elif importance == 2:
                weight_env, weight_budget = 0.3, 0.7
            else:
                weight_env, weight_budget = 0.5, 0.5

        total_w = max(weight_env + weight_budget, 1e-9)
        w_env = float(weight_env) / total_w
        w_budget = float(weight_budget) / total_w

        # ---------------- Build feature tokens ----------------
        # For each row, aggregate text fields to tokens
        text_cols = [c for c in ["tags", "features", "type", "location", "title", "name"] if c in df.columns]
        df["_agg_tokens"] = df[text_cols].astype(str).apply(lambda row: _tokenize(" ".join(row.values)), axis=1)

        # ---------------- Environment soft score (Jaccard) ----------------
        env_raw = getattr(user, "environment", getattr(user, "preferred_environment", ""))
        # Allow list or string
        if isinstance(env_raw, (list, tuple, set)):
            user_env_tokens = _to_set([t for v in env_raw for t in _tokenize(v)])
        else:
            user_env_tokens = _to_set(_tokenize(env_raw))

        def env_score(tokens: List[str]) -> float:
            if not user_env_tokens:
                return 0.5  # neutral if no environment preference
            s = _to_set(tokens)
            inter = len(user_env_tokens & s)
            union = len(user_env_tokens | s)
            return inter / union if union else 0.0

        df["env_match"] = df["_agg_tokens"].apply(env_score).astype(float)

        # ---------------- Budget score (exp kernel) ----------------
        prices = pd.to_numeric(df["nightly_price"], errors="coerce").astype(float)
        df["nightly_price"] = prices

        budget_min = getattr(user, "budget_min", None)
        budget_max = getattr(user, "budget_max", None)
        if budget_min is None and budget_max is None:
            single_budget = getattr(user, "budget", None)
            if single_budget is not None:
                budget_min = budget_max = float(single_budget)

        if budget_min is not None and budget_max is not None and budget_max >= budget_min:
            avg_budget = (float(budget_min) + float(budget_max)) / 2.0
            half_range = max((float(budget_max) - float(budget_min)) / 2.0, 1e-6)
            # Exponential kernel centered at avg_budget; ~1 near center, decays smoothly
            df["budget_score"] = np.exp(-((prices - avg_budget) / half_range) ** 2)
            # Mild penalty if outside hard band
            outside = (prices < float(budget_min)) | (prices > float(budget_max))
            df.loc[outside, "budget_score"] *= 0.7
        else:
            # Fallback: robust z-score using median/IQR of the dataset
            med = float(np.nanmedian(prices))
            iqr = float(np.nanpercentile(prices, 75) - np.nanpercentile(prices, 25))
            scale = max(iqr / 1.349, 1e-6)  # IQR->sigma approx for normal
            df["budget_score"] = np.exp(-((prices - med) / scale) ** 2)

        # ---------------- Capacity & constraints penalties (optional) ----------------
        # If dataset has "capacity" or "sleeps" and user has group_size, penalize infeasible
        group_size = getattr(user, "group_size", None)
        cap_col = "capacity" if "capacity" in df.columns else ("sleeps" if "sleeps" in df.columns else None)
        if group_size is not None and cap_col is not None:
            infeasible = df[cap_col].astype(float) < float(group_size)
            df.loc[infeasible, "env_match"] *= 0.8
            df.loc[infeasible, "budget_score"] *= 0.6  # stronger penalty

        # ---------------- Keyword BOOST from query/amenities ----------------
        boost = 0.0
        tokens = []
        if isinstance(query, str) and query.strip():
            tokens = [t for t in _tokenize(query) if t]
            if tokens:
                def count_matches(tok_list: List[str]) -> int:
                    s = _to_set(tok_list)
                    return sum(1 for t in tokens if t in s)
                df["_kw_matches"] = df["_agg_tokens"].apply(count_matches)
                # +0.1 per matched token (cap at +0.3)
                df["_kw_boost"] = (0.1 * df["_kw_matches"]).clip(upper=0.3)
                boost = None  # unused variable; per-row boost is in _kw_boost
            else:
                df["_kw_boost"] = 0.0
        else:
            df["_kw_boost"] = 0.0

        # ---------------- Final score ----------------
        df["fit_score"] = (w_env * df["env_match"] + w_budget * df["budget_score"] + df["_kw_boost"]).astype(float)

        # ---------------- Sort & format ----------------
        sort_keys = [("fit_score", False)]
        if "nightly_price" in df.columns:
            sort_keys.append(("nightly_price", True))
        if "name" in df.columns:
            sort_keys.append(("name", True))

        df_sort = df.sort_values(
            by=[k for k, _ in sort_keys],
            ascending=[asc for _, asc in sort_keys],
            kind="mergesort",
        ).head(self.top_k).reset_index(drop=True)

        # Round numeric columns for display
        for col in ["nightly_price", "fit_score", "env_match", "budget_score"]:
            if col in df_sort.columns:
                df_sort[col] = pd.to_numeric(df_sort[col], errors="coerce")
        if "nightly_price" in df_sort.columns:
            df_sort["nightly_price"] = df_sort["nightly_price"].round(2)
        for col in ["fit_score", "env_match", "budget_score"]:
            if col in df_sort.columns:
                df_sort[col] = df_sort[col].round(3)

        # === DISPLAY BLOCK (includes 'location') ===
        display_cols = [
            "location",
            "type",
            "nightly_price",
            "features",
            "fit_score",
            "env_match",
            "budget_score",
            "name",
            "title",
            "tags",
        ]
        display_cols = [c for c in display_cols if c in df_sort.columns]
        if self.echo_table and display_cols:
            display = df_sort[display_cols].copy()
            print("\nTop Recommendations\n" + "-" * 21)
            print(display.to_string(index=False))
            extra_info = []
            if tokens:
                extra_info.append(f"keyword boost up to +0.3 ({len(tokens)} token(s))")
            extra_info.append(f"weights -> env: {w_env:.2f}, price: {w_budget:.2f}")
            print("(" + "; ".join(extra_info) + ")")

        # Clean temp columns before returning
        for c in ["_agg_tokens", "_kw_matches", "_kw_boost"]:
            if c in df_sort.columns:
                pass  # they won't exist in df_sort unless copied; safe to ignore
        return df_sort
