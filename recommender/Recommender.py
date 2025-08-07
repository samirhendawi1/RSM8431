class Recommender:
    def recommend(self, user, properties_df):
        df = properties_df.copy()

        df['env_match'] = df['tags'].str.contains(user.environment, case=False, na=False).astype(int)
        df['budget_score'] = 1 - abs((df['nightly_price'] - ((user.budget_min + user.budget_max) / 2)) / ((user.budget_max - user.budget_min) + 1e-5))
        df['fit_score'] = 0.6 * df['env_match'] + 0.4 * df['budget_score']

        return df.sort_values(by='fit_score', ascending=False).head(5)

