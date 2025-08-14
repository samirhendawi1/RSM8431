from user.UserManager import UserManager
from properties.PropertyManager import PropertyManager, generate_properties_csv
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper
from smart_search import SmartSearch
import os
from pathlib import Path
from datetime import datetime

# ---- additions for error handling ----
import builtins
import sys

def say_error(msg, err=None):
    try:
        if err:
            print(f"[error] {msg}: {err}")
        else:
            print(f"[error] {msg}")
    except Exception:
        print("[error] unexpected printing failure")

_original_input = builtins.input
def ask_input_safely(prompt: str):
    try:
        if prompt == "Choose an option: ":
            valid = {"1","2","3","4","5","6","7","8"}
            while True:
                try:
                    ans = _original_input(prompt)
                except (EOFError, KeyboardInterrupt):
                    print("[info] exiting")
                    return "8"
                if ans is None:
                    continue
                ans = str(ans).strip()
                if ans in valid:
                    return ans
                print("Please enter a number from 1 to 8.")
        elif prompt.startswith("Type DELETE to remove profile"):
            while True:
                try:
                    ans = _original_input(prompt)
                except (EOFError, KeyboardInterrupt):
                    print("[info] delete cancelled")
                    return "cancel"
                if ans is None:
                    continue
                s = str(ans).strip()
                if s == "DELETE" or s.lower() == "cancel":
                    return s
                print("Please type exactly DELETE, or 'cancel' to stop.")
        else:
            return _original_input(prompt)
    except (EOFError, KeyboardInterrupt):
        print("[info] input cancelled")
        return ""
builtins.input = ask_input_safely

_original_generate_properties_csv = generate_properties_csv
def generate_properties_csv(path: str, n: int):
    try:
        return _original_generate_properties_csv(path, n)
    except Exception as e:
        say_error("could not generate demo properties; continuing", e)
globals()["generate_properties_csv"] = generate_properties_csv

try:
    _orig_llm_blurb = LLMHelper.generate_travel_blurb
    def generate_travel_blurb(self, prompt: str):
        try:
            return _orig_llm_blurb(self, prompt)
        except Exception as e:
            say_error("llm blurb failed", e)
            return "Sorry, I couldn't generate a blurb right now."
    LLMHelper.generate_travel_blurb = generate_travel_blurb

    if hasattr(LLMHelper, "search"):
        _orig_llm_search = LLMHelper.search
        def search(self, query: str):
            try:
                return _orig_llm_search(self, query)
            except Exception as e:
                say_error("llm search failed", e)
                return {"tags": [], "property_ids": []}
        LLMHelper.search = search
except Exception as e:
    say_error("llm wrappers setup failed", e)

def student_excepthook(exc_type, exc, tb):
    print(f"[error] unexpected issue: {exc_type.__name__}: {exc}")
sys.excepthook = student_excepthook

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
    
        # Validate budgets before recommending
        try:
            bmin = float(getattr(user, "budget_min", 0))
            bmax = float(getattr(user, "budget_max", 0))
            if bmin <= 0 or bmax <= 0:
                print("Invalid budget(s). Please edit your profile to set positive budgets.")
                continue
            if bmin > bmax:
                print("Invalid budget range (min > max). Please edit your profile.")
                continue
        except Exception:
            print("Budget values look invalid. Please edit your profile.")
            continue
    
        # Optional freeform query + LLM assist
        freeform = input("Add any tags, features, or describe the property (optional): ").strip()
        ss = SmartSearch(property_manager.properties)
        candidates = property_manager.properties
        llm_tags, llm_ids = [], []
    
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
    
        # Boost candidates if LLM gave tags/IDs
        llm_tags = ss.canonicalize_tags(llm_tags)
        textcol = "__fulltext__" if "__fulltext__" in candidates.columns else None
        if llm_tags and textcol:
            has_tag = candidates[textcol].fillna("").astype(str).apply(
                lambda t: any(tag in t for tag in llm_tags)
            )
            candidates = candidates.copy()
            candidates["semantic_score"] = candidates.get("semantic_score", 0.0) + has_tag.astype(float) * 0.15
    
        if llm_ids and "property_id" in candidates.columns:
            candidates = candidates.copy()
            candidates["semantic_score"] = candidates.get("semantic_score", 0.0) + \
                                           candidates["property_id"].isin(llm_ids).astype(float) * 0.20
    
        # Recommend + export
        try:
            top5 = recommender.recommend(user, candidates)
            export_dir = Path("output")
            export_dir.mkdir(parents=True, exist_ok=True)
    
            def _safe_filename(s):
                s = s or ""
                return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")
    
            user_suffix = _safe_filename(getattr(user, "name", None))
            fname = f"recommendations_{user_suffix}.csv" if user_suffix else "recommendations.csv"
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

                # FIX: remove invalid isinstance(user_manager, 'users'); operate on dict safely
                users_dict = getattr(user_manager, 'users', None)
                if isinstance(users_dict, dict):
                    if uid in users_dict:
                        del users_dict[uid]
                    else:
                        for k, v in list(users_dict.items()):
                            if v is user:
                                del users_dict[k]

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
