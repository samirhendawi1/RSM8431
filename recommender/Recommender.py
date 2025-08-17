import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union


class Recommender:

    def __init__(self, top_k: int = 5, echo_table: bool = True):
        self.top_k = top_k
        self.echo_table = echo_table

    def _as_lower_str(self, s: Optional[str]) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        return str(s).lower()

    # NOTE: flexible signature to avoid "takes 3 positional args but 4 were given"
    def recommend(
            self,
            user,
            properties_df: pd.DataFrame,
            extra: Optional[Union[int, float, str, Tuple[float, float], List[float]]] = None,
            **kwargs,
    ) -> pd.DataFrame:
        """
        extra/kwargs compatibility:
          - importance: 1 (env) | 2 (price) | 3 (equal)
          - query: string of keywords to filter rows on (tags/features/type/location)
          - weight_env, weight_budget: explicit weights
        """
        if properties_df is None or len(properties_df) == 0:
            return pd.DataFrame()

        df = properties_df.copy()

        # Normalize column names commonly seen in prior versions
        if "type" not in df.columns and "ptype" in df.columns:
            df["type"] = df["ptype"]

        # Basic required columns
        required = ["nightly_price"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Recommender: missing required column(s): {missing}")

        # --------- Parse compatibility args ---------
        # Defaults
        importance = kwargs.pop("importance", None)
        query = kwargs.pop("query", None)
        weight_env = kwargs.pop("weight_env", None)
        weight_budget = kwargs.pop("weight_budget", None)

        # Interpret 'extra' if provided
        if extra is not None:
            if isinstance(extra, (tuple, list)) and len(extra) >= 2:
                weight_env = float(extra[0])
                weight_budget = float(extra[1])
            elif isinstance(extra, (int, float)):
                # treat as importance choice if in {1,2,3}
                if int(extra) in (1, 2, 3):
                    importance = int(extra)
                else:
                    # ignore unusual numeric 'extra'
                    pass
            elif isinstance(extra, str):
                # If it's a digit-like string, treat as importance, else as query
                if extra.strip().isdigit():
                    val = int(extra.strip())
                    if val in (1, 2, 3):
                        importance = val
                else:
                    query = extra

        # If user gave explicit weights, use them. Else map 'importance' to weights.
        if weight_env is None or weight_budget is None:
            if importance == 1:
                # Environment prioritized
                weight_env, weight_budget = 0.7, 0.3
            elif importance == 2:
                # Price prioritized
                weight_env, weight_budget = 0.3, 0.7
            else:
                # equal or unspecified
                weight_env, weight_budget = 0.5, 0.5

        # Normalize positive weights
        total_w = max(weight_env + weight_budget, 1e-9)
        w_env = float(weight_env) / total_w
        w_budget = float(weight_budget) / total_w

        # Optional keyword filter
        if isinstance(query, str) and query.strip():
            tokens = [t.strip().lower() for t in query.split() if t.strip()]
            if tokens:
                hay_cols = [c for c in ["tags", "features", "type", "location", "title", "name"] if c in df.columns]
                def row_matches(row):
                    hay = " ".join([self._as_lower_str(row.get(c, "")) for c in hay_cols])
                    return all(tok in hay for tok in tokens)
                df = df[df.apply(row_matches, axis=1)]
                if df.empty:
                    # No matches; fall back to original (but keep doing the scoring)
                    df = properties_df.copy()

        # --------- Scoring ---------
        tags = df.get("tags", "").fillna("").astype(str).map(self._as_lower_str).values
        prices = pd.to_numeric(df["nightly_price"], errors="coerce").fillna(np.inf).values

        # Preference vector (environment match)
        env_pref = self._as_lower_str(getattr(user, "environment", getattr(user, "preferred_environment", "")))
        env_match = np.array([1 if (env_pref and env_pref in t) else 0 for t in tags], dtype=float)

        # Budget score (closer to midpoint of the user's range is better)
        budget_min = getattr(user, "budget_min", None)
        budget_max = getattr(user, "budget_max", None)
        if budget_min is None and budget_max is None:
            single_budget = getattr(user, "budget", None)
            if single_budget is not None:
                budget_min = budget_max = float(single_budget)

        if budget_min is None or budget_max is None:
            budget_score = np.ones_like(prices, dtype=float) * 0.5
        else:
            avg_budget = (float(budget_min) + float(budget_max)) / 2.0
            range_budget = max(float(budget_max) - float(budget_min), 1e-5)
            budget_score = 1.0 - np.abs(prices - avg_budget) / range_budget
            budget_score = np.clip(budget_score, 0.0, 1.0)

        # Final fit score (weights now driven by importance or explicit)
        fit_score = w_env * env_match + w_budget * budget_score

        # Attach scores
        df["env_match"] = env_match
        df["budget_score"] = budget_score
        df["fit_score"] = fit_score

        # Sort by fit_score descending; break ties by lower price then by name if available
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

        # Round numeric columns for nicer display
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
            print(f"(Weights -> env: {w_env:.2f}, price: {w_budget:.2f}. "
                  "Scores: env_match/budget_score in [0,1]; fit = w_env*env + w_budget*budget)")

        return df_sort
