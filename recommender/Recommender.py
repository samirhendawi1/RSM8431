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
            budget_min = float(getattr(user, "budget_min", 0) or 0)
            budget_max = float(getattr(user, "budget_max", 0) or 0)
            avg_budget = (budget_min + budget_max) / 2.0
            price_range = (budget_max - budget_min) + 1e-5
            df["budget_score"] = 1 - abs((pd.to_numeric(df["nightly_price"], errors="coerce").fillna(avg_budget) - avg_budget) / price_range)
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
