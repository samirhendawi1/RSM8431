from user.UserManager import UserManager
from properties.PropertyManager import PropertyManager, generate_properties_csv
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper
from smart_search import SmartSearch
import os
from pathlib import Path
from datetime import datetime

# ---------- Helpers ----------
def _derive_available_envs(df):
    """Infer available 'environments' from tags/type/environment columns."""
    envs = set()
    dd = df.copy()
    dd.columns = [c.lower() for c in dd.columns]

    if "tags" in dd.columns:
        tags_series = dd["tags"].fillna("").astype(str).str.lower()
        for parts in tags_series.str.split(","):
            for t in parts:
                t = t.strip()
                if t:
                    envs.add(t)

    if "type" in dd.columns:
        envs |= set(dd["type"].fillna("").astype(str).str.lower().unique().tolist())

    if "environment" in dd.columns:
        envs |= set(dd["environment"].fillna("").astype(str).str.lower().unique().tolist())

    return {e for e in envs if e and e != "nan"}


def _normalize_environment(raw_env, ss: SmartSearch, available_envs):
    """Map user-entered environment to the closest dataset environment."""
    raw = (raw_env or "").strip().lower()
    if not raw:
        return raw_env
    toks = ss.canonicalize_tags([raw])  # alias expansion
    candidates = [raw] + toks
    for cand in candidates:
        if cand in available_envs:
            return cand
    for cand in available_envs:
        if raw in cand:
            return cand
    return raw_env


def _print_profile(user):
    print("\n--- Current User Profile ---")
    print(f"ID:            {getattr(user, 'user_id', 'N/A')}")
    print(f"Name:          {getattr(user, 'name', 'N/A')}")
    print(f"Group size:    {getattr(user, 'group_size', 'N/A')}")
    print(f"Environment:   {getattr(user, 'environment', 'N/A')}")
    print(f"Budget min:    {getattr(user, 'budget_min', 'N/A')}")
    print(f"Budget max:    {getattr(user, 'budget_max', 'N/A')}")
    print(f"Travel dates:  {getattr(user, 'travel_dates', 'N/A')}")
    print("-----------------------------\n")


# ---------- Main ----------
def main():
    # Generate demo data (keep as per your flow)
    generate_properties_csv("data/properties.csv", 100)

    user_manager = UserManager()
    property_manager = PropertyManager('data/properties.csv')
    recommender = Recommender()
    # Short timeout; skip LLM if no API key
    llm = LLMHelper(csv_path="data/properties_expanded.csv", request_timeout=6)
    ss = SmartSearch(property_manager.properties)

    available_envs = _derive_available_envs(property_manager.properties)

    while True:
        print("\nMenu:")
        print("1. Create User")
        print("2. Edit Profile")
        print("3. View Profile")
        print("4. Show Properties")
        print("5. Get Recommendations (exports CSV)")
        print("6. Generate a blurb via a LLM")
        print("7. Delete Profile")
        print("8. Exit")

        choice = input("Choose an option: ").strip()

        if choice == '1':
            if hasattr(user_manager, 'create_user'):
                user_manager.create_user()
                user = getattr(user_manager, 'get_current_user', lambda: None)()
                if user and getattr(user, 'environment', None):
                    old_env = user.environment
                    new_env = _normalize_environment(old_env, ss, available_envs)
                    if new_env != old_env:
                        user.environment = new_env
                        print(f"(Adjusted environment: '{old_env}' → '{new_env}' based on dataset)")
            else:
                print("Create user is not available in UserManager.")
        elif choice == '2':
            if hasattr(user_manager, 'edit_profile'):
                user_manager.edit_profile()
                user = getattr(user_manager, 'get_current_user', lambda: None)()
                if user and getattr(user, 'environment', None):
                    old_env = user.environment
                    new_env = _normalize_environment(old_env, ss, available_envs)
                    if new_env != old_env:
                        user.environment = new_env
                        print(f"(Adjusted environment: '{old_env}' → '{new_env}' based on dataset)")
            else:
                print("Edit profile is not available in UserManager.")
        elif choice == '3':
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if user:
                _print_profile(user)
            else:
                print("No user selected.")
        elif choice == '4':
            if hasattr(property_manager, 'display_properties'):
                property_manager.display_properties()
            else:
                print("Property display is not available.")
        elif choice == '5':
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if user is None:
                print("No user selected.")
                continue

            freeform = input("Add any tags, features, or describe the property (optional): ").strip()

            llm_tags, llm_ids = [], []
            candidates = property_manager.properties

            if freeform:
                candidates = ss.find_candidates(freeform, top_k=300)

                if os.getenv("OPENROUTER_API_KEY"):
                    try:
                        resp = llm.search(freeform)
                        if isinstance(resp, dict) and "error" not in resp:
                            llm_tags = resp.get("tags", []) or []
                            llm_ids  = resp.get("property_ids", []) or []
                    except Exception:
                        pass

                # Canonicalize and boost
                llm_tags = ss.canonicalize_tags(llm_tags)
                textcol = "__fulltext__" if "__fulltext__" in candidates.columns else None
                if llm_tags and textcol:
                    has_tag = candidates[textcol].apply(lambda t: any(tag in t for tag in llm_tags))
                    candidates = candidates.copy()
                    candidates["semantic_score"] = candidates.get("semantic_score", 0.0) + has_tag.astype(float) * 0.15

                if llm_ids and "property_id" in candidates.columns:
                    candidates = candidates.copy()
                    candidates["semantic_score"] = candidates.get("semantic_score", 0.0) + \
                                                   candidates["property_id"].isin(llm_ids).astype(float) * 0.20

            try:
                recs = recommender.recommend(user, candidates)

                top5 = recs.head(5)
                script_dir = Path(__file__).resolve().parent
                export_dir = script_dir / "exports"
                export_dir.mkdir(parents=True, exist_ok=True)

                user_suffix = getattr(user, "name", None)
                fname = f"recommendations{('_' + user_suffix) if user_suffix else ''}.csv"
                output_path = export_dir / fname

                preferred_cols = [c for c in ["location", "type", "nightly_price", "features", "fit_score"] if c in top5.columns]
                df_to_save = top5[preferred_cols] if preferred_cols else top5
                try:
                    import pandas as _pd
                    if not isinstance(df_to_save, _pd.DataFrame):
                        df_to_save = _pd.DataFrame(df_to_save)
                except Exception:
                    pass

                df_to_save.to_csv(output_path, index=False, encoding="utf-8")
                print(f"\nTop 5 recommendations saved to: {output_path}")

            except Exception as e:
                print("Error generating recommendations:", e)
        elif choice == '6':
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if user:
                prompt = (
                    f"Write a fun intro for a user looking for a {user.environment} stay "
                    f"with {user.group_size} friends under ${user.budget_max}/night."
                )
                print("\nLLM Response:")
                print(llm.generate_travel_blurb(prompt))
            else:
                print("No user selected.")
        elif choice == '7':
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if not user:
                print("No user selected.")
                continue
            confirm = input(f"Type DELETE to remove profile '{getattr(user, 'name', '')}': ").strip()
            if confirm != "DELETE":
                print("Cancelled.")
                continue
            try:
                uid = getattr(user, 'user_id', None)
                if hasattr(user_manager, 'users') and isinstance(user_manager, 'users'):
                    pass  # defensive
                if hasattr(user_manager, 'users') and isinstance(user_manager.users, dict):
                    if uid in user_manager.users:
                        del user_manager.users[uid]
                    else:
                        for k, v in list(user_manager.users.items()):
                            if v is user:
                                del user_manager.users[k]
                if getattr(user_manager, 'current_user_id', None) == uid:
                    user_manager.current_user_id = None
                print("Profile deleted.")
            except Exception as e:
                print("Failed to delete profile:", e)
        elif choice == '8':
            break
        else:
            print("Invalid option.")


if __name__ == '__main__':
    main()
