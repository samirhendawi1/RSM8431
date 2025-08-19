import os
from pathlib import Path

from user.UserManager import UserManager, User
from properties.PropertyManager import PropertyManager, generate_properties_csv
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper
from smart_search import SmartSearch


def _safe_filename(s: str) -> str:
    s = s or ""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")


def main():
    # Use the file you actually have in your repo
    data_path = "data/properties_with_capacity_types.csv"
    if not os.path.isfile(data_path):
        generate_properties_csv(data_path, 140)

    user_manager = UserManager()
    pm = PropertyManager(data_path)

    # Default recommender
    rec = Recommender(top_k=5, weights=(0.4, 0.4, 0.2), llm_weight=0.25)

    # SmartSearch pre-filter
    ss = SmartSearch(pm.properties)

    # LLM helper (works with env var or interactive prompt)
    try:
        llm = LLMHelper(csv_path=data_path, request_timeout=8)
    except Exception:
        llm = None

    while True:
        print("\nMenu:")
        print("1. Sign up")
        print("2. Sign in")
        print("3. Sign out")
        print("4. Edit profile")
        print("5. View profile")
        print("6. Show properties")
        print("7. Get recommendations (exports CSV)")
        print("8. Search and Filter Properties")
        print("9. Delete profile")
        print("0. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            user_manager.sign_up()

        elif choice == "2":
            user_manager.sign_in()

        elif choice == "3":
            user_manager.sign_out()

        elif choice == "4":
            # NOTE: edit_profile is restricted to username + password only (see UserManager.py)
            user_manager.edit_profile()

        elif choice == "5":
            # Minimal profile view (username only)
            user_manager.view_profile()

            # Show last recommendations for the current user, if any exist
            u = user_manager.get_current_user()
            if not u:
                continue

            # Prefer username file; ignore blank name to avoid recommendations_.csv
            username_candidate = Path("output") / f"recommendations_{_safe_filename(u.username)}.csv"
            name_candidate = None
            if getattr(u, "name", None):
                name_candidate = Path("output") / f"recommendations_{_safe_filename(u.name)}.csv"

            pick = None
            if username_candidate.is_file():
                pick = username_candidate
            elif name_candidate and name_candidate.is_file():
                pick = name_candidate

            if pick is None:
                # Fallback: latest file that contains username
                try:
                    outs = sorted(Path("output").glob("recommendations_*.csv"),
                                  key=lambda x: x.stat().st_mtime, reverse=True)
                    for p in outs:
                        if _safe_filename(u.username) in p.name:
                            pick = p
                            break
                except Exception:
                    pick = None

            if pick is None:
                print("No past recommendations found.")
            else:
                try:
                    import pandas as pd
                    df = pd.read_csv(pick)
                    print(f"Latest recommendations file: {pick}")
                    # Show ONLY user-friendly columns — NO scores
                    show_cols = [c for c in [
                        "property_id", "location", "environment", "property_type", "nightly_price",
                        "min_guests", "max_guests", "features", "tags"
                    ] if c in df.columns]
                    if not show_cols:
                        print("(No displayable columns found.)")
                    else:
                        print(df[show_cols].head(5).to_string(index=False))
                except Exception as e:
                    print(f"Could not read {pick}: {e}")

        elif choice == "6":
            pm.display_properties()

        elif choice == "7":
            user = user_manager.get_current_user()
            if not user:
                print("Sign in first.")
                continue

            print("\n-- Recommendation Inputs (blank keeps default) --")
            loc = input("Location contains (optional): ").strip()
            env_in = input(f"Environment [{user.environment or 'none'}]: ").strip().lower() or user.environment
            gs_in = input(f"Group size [{user.group_size}]: ").strip()
            bmin_in = input(f"Budget min [{user.budget_min}]: ").strip()
            bmax_in = input(f"Budget max [{user.budget_max}]: ").strip()

            try:
                group_size = int(gs_in) if gs_in else int(getattr(user, "group_size", 0) or 0)
            except Exception:
                group_size = int(getattr(user, "group_size", 0) or 0)

            try:
                budget_min = float(bmin_in) if bmin_in else float(getattr(user, "budget_min", 0) or 0)
            except Exception:
                budget_min = float(getattr(user, "budget_min", 0) or 0)

            try:
                budget_max = float(bmax_in) if bmax_in else float(getattr(user, "budget_max", 0) or 0)
            except Exception:
                budget_max = float(getattr(user, "budget_max", 0) or 0)

            if budget_min > budget_max:
                print("Budget min cannot exceed budget max. Swapping.")
                budget_min, budget_max = budget_max, budget_min

            freeform = input("\nDescribe preferences (optional free-text): ").strip()
            candidates = pm.properties

            if loc:
                try:
                    candidates = candidates[candidates["location"].str.lower().str.contains(loc.lower(), na=False)]
                except Exception:
                    pass

            llm_tags, llm_ids, blurb = [], [], None

            if freeform:
                try:
                    ss = SmartSearch(candidates)
                    candidates = ss.find_candidates(freeform, top_k=300)
                except Exception:
                    pass

                if llm and getattr(llm, "api_key", ""):
                    try:
                        resp = llm.search(freeform)
                        if isinstance(resp, dict):
                            llm_tags = resp.get("tags", []) or []
                            llm_ids = resp.get("property_ids", []) or []
                    except Exception as e:
                        print("LLM search error:", e)

                    try:
                        blurb_resp = llm.generate_travel_blurb(freeform)
                        if isinstance(blurb_resp, str) and not blurb_resp.startswith("ERROR:"):
                            blurb = blurb_resp
                        elif isinstance(blurb_resp, str) and blurb_resp.startswith("ERROR:"):
                            print("LLM blurb error:", blurb_resp)
                    except Exception as e:
                        print("LLM blurb exception:", e)

            tmp_user = User(
                username=getattr(user, "username", ""),
                name=getattr(user, "name", ""),
                group_size=group_size,
                environment=env_in,
                budget_min=budget_min,
                budget_max=budget_max,
            )

            top5 = rec.recommend(tmp_user, candidates, llm_tags=llm_tags, llm_ids=llm_ids)

            Path("output").mkdir(parents=True, exist_ok=True)
            fname = f"recommendations_{_safe_filename(user.name or user.username)}.csv"
            out = Path("output") / fname
            try:
                top5.to_csv(out, index=False, encoding="utf-8")
            except Exception as e:
                print(f"Failed to save CSV: {e}")

            if blurb:
                print("\n— Blurb —")
                print(blurb)

            print(f"\nTop 5 saved to: {out}")
            # Display WITHOUT any score columns
            show_cols = [c for c in [
                "property_id", "location", "environment", "property_type", "nightly_price",
                "min_guests", "max_guests", "features", "tags"
            ] if c in top5.columns]
            try:
                print(top5[show_cols].to_string(index=False))
            except Exception:
                print(top5[show_cols].head(5))

        elif choice == "8":
            d = pm.properties
            print(f"\nRows in dataset: {len(d)}")
            loc = input("Location contains (or blank): ").strip().lower()
            if loc:
                d = d[d["location"].str.lower().str.contains(loc, na=False)]
            typ = input("Property type contains (or blank): ").strip().lower()
            if typ:
                d = d[d["property_type"].str.lower().str.contains(typ, na=False)]
            pmin = input("Min price (or blank): ").strip()
            pmax = input("Max price (or blank): ").strip()
            try:
                if pmin:
                    d = d[d["nightly_price"] >= float(pmin)]
                if pmax:
                    d = d[d["nightly_price"] <= float(pmax)]
            except Exception:
                pass
            gs = input("Group size (or blank): ").strip()
            try:
                if gs:
                    g = int(gs)
                    d = d[(d["min_guests"] <= g) & (g <= d["max_guests"])]
            except Exception:
                pass
            print("\nResults:")
            show = [c for c in ["property_id", "location", "property_type", "nightly_price",
                                "min_guests", "max_guests", "features", "tags"] if c in d.columns]
            print(d[show].head(50).to_string(index=False))

        elif choice == "9":
            user_manager.delete_user()

        elif choice == "0":
            print("Bye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
