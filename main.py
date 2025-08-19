import os
from pathlib import Path

from user.UserManager import UserManager
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

    # Recommender: env, budget, group; llm_weight tunes LLM influence
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
            user_manager.edit_profile()

        elif choice == "5":
            # Minimal profile view (username only)
            user_manager.view_profile()

            # Show last 5 recommendations for the current user, if any exist
            u = user_manager.get_current_user()
            if not u:
                continue

            # Try both filename patterns: by 'name' (legacy) and by 'username' (fallback)
            def _safe_filename(s):
                s = s or ""
                return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s).strip("_")

            candidates = [
                Path("output") / f"recommendations_{_safe_filename(u.name)}.csv",
                Path("output") / f"recommendations_{_safe_filename(u.username)}.csv",
                ]
            latest = None
            for p in candidates:
                if p.is_file():
                    latest = p
                    break

            if latest is None:
                # If nothing matched, try to find any file that includes the username
                try:
                    outs = sorted(Path("output").glob("recommendations_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
                    for p in outs:
                        if u.username in p.name or _safe_filename(u.name) in p.name:
                            latest = p
                            break
                except Exception:
                    latest = None

            if latest is None:
                print("No past recommendations found.")
            else:
                try:
                    import pandas as pd
                    df = pd.read_csv(latest)
                    print(f"Latest recommendations file: {latest}")
                    # Show the first 5 rows with useful columns if present
                    preferred = [c for c in [
                        "property_id","location","environment","property_type","nightly_price",
                        "min_guests","max_guests","features","tags","fit_score"
                    ] if c in df.columns]
                    print(df[preferred].head(5).to_string(index=False))
                except Exception as e:
                    print(f"Could not read {latest}: {e}")

        elif choice == "6":
            pm.display_properties()

        elif choice == "7":
            user = user_manager.get_current_user()
            if not user:
                print("Sign in first.")
                continue

            freeform = input("Describe preferences (optional free-text): ").strip()
            candidates = pm.properties
            llm_tags, llm_ids = [], []
            blurb = None

            # Use SmartSearch when user types something
            if freeform:
                try:
                    candidates = ss.find_candidates(freeform, top_k=300)
                except Exception:
                    candidates = pm.properties

                # IMPORTANT: gate on llm.api_key (works for env OR getpass)
                if llm and getattr(llm, "api_key", ""):
                    # Tags / IDs for scoring
                    try:
                        resp = llm.search(freeform)
                        if isinstance(resp, dict):
                            llm_tags = resp.get("tags", []) or []
                            llm_ids = resp.get("property_ids", []) or []
                    except Exception as e:
                        print("LLM search error:", e)

                    # Short blurb for the result header
                    try:
                        blurb_resp = llm.generate_travel_blurb(freeform)
                        if isinstance(blurb_resp, str) and not blurb_resp.startswith("ERROR:"):
                            blurb = blurb_resp
                        else:
                            # Surface the error once so you know what's wrong
                            if isinstance(blurb_resp, str) and blurb_resp.startswith("ERROR:"):
                                print("LLM blurb error:", blurb_resp)
                    except Exception as e:
                        print("LLM blurb exception:", e)

            # Rank & export
            top5 = rec.recommend(user, candidates, llm_tags=llm_tags, llm_ids=llm_ids)

            Path("output").mkdir(parents=True, exist_ok=True)
            fname = f"recommendations_{_safe_filename(user.name)}.csv"
            out = Path("output") / fname
            top5.to_csv(out, index=False, encoding="utf-8")

            # Print blurb then table
            if blurb:
                print("\n— Blurb —")
                print(blurb)

            print(f"\nTop 5 saved to: {out}")
            print(top5.to_string(index=False))

        elif choice == "8":
            # Simple filter utility
            d = pm.properties
            print(f"\nRows in dataset: {len(d)}")
            loc = input("Location contains (or blank): ").strip().lower()
            if loc:
                d = d[d["location"].str.lower().str.contains(loc, na=False)]
            typ = input("Property type contains (or blank): ").strip().lower()
            if typ:
                # property_type is the correct column name in your dataset
                d = d[d["property_type"].str.lower().str.contains(typ, na=False)]
            pmin = input("Min price (or blank): ").strip()
            pmax = input("Max price (or blank): ").strip()
            try:
                if pmin: d = d[d["nightly_price"] >= float(pmin)]
                if pmax: d = d[d["nightly_price"] <= float(pmax)]
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
            show = [c for c in ["property_id","location","property_type","nightly_price",
                                "min_guests","max_guests","features","tags"] if c in d.columns]
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
