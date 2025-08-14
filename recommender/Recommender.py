class Recommender:
    def recommend(self, user, properties_df):
        try:
            df = properties_df.copy()
        except Exception:
            # If it's not a DataFrame, try to coerce
            import pandas as pd
            try:
                df = pd.DataFrame(properties_df)
            except Exception:
                # Return empty DataFrame if coercion fails
                return properties_df

        # Ensure required columns exist
        for col in ["tags", "nightly_price"]:
            if col not in df.columns:
                df[col] = ""

        # Environment match score (safe even if user.environment is None/empty)
        try:
            env = getattr(user, "environment", "") or ""
            df["env_match"] = df["tags"].astype(str).str.contains(str(env), case=False, na=False).astype(int)
        except Exception:
            df["env_match"] = 0

        # Budget score (safe even if missing/invalid numbers)
        try:
            import pandas as pd
            # nightly_price → numeric
            df["nightly_price"] = pd.to_numeric(df["nightly_price"], errors="coerce")

            bmin = float(getattr(user, "budget_min", 0))
            bmax = float(getattr(user, "budget_max", 0))
            avg_budget = (bmin + bmax) / 2.0

            if bmin <= 0 or bmax <= 0 or bmax < bmin:
                # invalid budgets → flat zero
                df["budget_score"] = 0.0
            else:
                price_range = max(abs(bmax - bmin), 1.0)  # avoid division by zero
                df["budget_score"] = 1 - (
                    df["nightly_price"].fillna(avg_budget).sub(avg_budget).abs() / price_range
                )
                df["budget_score"] = df["budget_score"].clip(lower=0.0, upper=1.0)
        except Exception:
            df["budget_score"] = 0.0

        # Fit score
        try:
            df["fit_score"] = 0.6 * df["env_match"] + 0.4 * df["budget_score"]
        except Exception:
            df["fit_score"] = 0.0

        try:
            return df.sort_values(by="fit_score", ascending=False).head(5)
        except Exception:
            return df.head(5)
