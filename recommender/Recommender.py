class Recommender:
    def recommend(self, user, properties_df, choices):
        # 1) Coerce to DataFrame if needed
        try:
            df = properties_df.copy()
        except Exception:
            import pandas as pd
            try:
                df = pd.DataFrame(properties_df)
            except Exception:
                return properties_df

        # 2) Ensure required columns exist
        for col in ["tags", "nightly_price"]:
            if col not in df.columns:
                df[col] = ""

        # 3) Environment match score
        env = getattr(user, "environment", "") or ""
        if env:
            df["env_match"] = (
                df["tags"].astype(str).str.contains(env, case=False, na=False, regex = False).astype(int)
            )
        #except Exception:
        else:
            df["env_match"] = 0

        # 4) Budget score
        try:
            import pandas as pd
            df["nightly_price"] = pd.to_numeric(df["nightly_price"], errors="coerce")

            bmin = float(getattr(user, "budget_min", 0))
            bmax = float(getattr(user, "budget_max", 0))
            avg_budget = (bmin + bmax) / 2.0

            if bmin <= 0 or bmax <= 0 or bmax < bmin:
                df["budget_score"] = 0.0
            else:
                price_range = max(abs(bmax - bmin), 1.0)  # avoid div-by-zero
                df["budget_score"] = 1 - (
                    df["nightly_price"].fillna(avg_budget).sub(avg_budget).abs() / price_range
                )
                df["budget_score"] = df["budget_score"].clip(lower=0.0, upper=1.0)
            df["budget_score"] = df["budget_score"].round(3)
        except Exception:
            df["budget_score"] = 0.0

        # 5) Fit score
        try:
            if choices == '1':
                w_env, w_budget = 0.7, 0.3
            elif choices == '2':
                w_env, w_budget = 0.3, 0.7
            else: #equal weight if no preference is given
                w_env, w_budget = 0.5, 0.5

            df["fit_score"] = (w_env * df["env_match"] + w_budget * df["budget_score"]) * 5
            df["fit_score"] = df["fit_score"].round(3)
        except Exception:
            df["fit_score"] = 0.0

        # 6) Sort, round numerics, and print a clean table
        try:
            df_sort = df.sort_values(by="fit_score", ascending=False).head(5).copy()

            # Choose display columns if present
            display_cols = ["name", "title", "tags", "nightly_price",
                                 "env_match", "budget_score", "fit_score"]
            display_cols = [c for c in display_cols if c in df_sort.columns]
            if not display_cols: 
                display_cols = list(df_sort.columns)

            display = df_sort[display_cols].copy()

            print("\nTop Recommendations\n" + "-" * 21)
            print(display.to_string(index=False))
            print("Note: Fit scores are out of 5.0")

            return df_sort
        except Exception:
            return df.head(5)
