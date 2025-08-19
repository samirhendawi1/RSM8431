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

    # Default recommender (weights fixed; no tuning UI)
    rec = Recommender(top_k=5, weights=(0.4, 0.4, 0.2), llm_weight=0.25)

    # SmartSearch pre-filter
    ss = SmartSearch(pm.properties)

    # LLM helper (uses env var or interactive prompt)
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
            # Restricted to username/password/first-name only (implemented in UserManager)
            user_manager.edit_profile()

        elif choice == "5":
            # Minimal profile view (username + first name)
            user_manager.view_profile()

            # Show last recommendations for the current user, if any exist
            u = user_manager.get_current_user()
            if not u:
                continue

            # Prefer username-based file; avoid blank-name files like recommendations_.csv
            username_candidate = Path("output") / f"recommendations_{_safe_filename(u.username)}.csv"
            name_candidate = Path("output") / f"recommendations_{_safe_filename(u.name)}.csv" if getattr(u, "name", None) else None

            pick = None
            if username_candidate.is_file():
                pick = username_candidate
            elif name_candidate and name_candidate.is_file():
                pick = name_candidate
            else:
                # Fallback: latest recommendations_* that contains the username
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

            # ---- Collect per-run preferences (session-only; NOT saved) ----
            print("\n-- Recommendation Inputs (blank keeps default) --")
            loc = input("Location contains (optional): ").strip()
            env_in = input(f"Environment [{user.environment or 'none'}]: ").strip().lower() or user.environment
            gs_in = input(f"Group size [{user.group_size}]: ").strip()
            bmin_in = input(f"Budget min [{user.budget_min}]: ").strip()
            bmax_in = input(f"Budget max [{user.budget_max}]: ").strip()

            # Safe parsing with fallbacks (still not persisted)
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

            # Start with full dataset; apply location pre-filter if given
            candidates = pm.properties
            if loc:
                try:
                    candidates = candidates[candidates["location"].str.lower().str.contains(loc.lower(), na=False)]
                except Exception:
                    pass  # fail open

            llm_tags, llm_ids = [], []
            # Use SmartSearch to narrow candidates when freeform provided
            if freeform:
                try:
                    ss_local = SmartSearch(candidates)
                    candidates = ss_local.find_candidates(freeform, top_k=300)
                except Exception:
                    pass

                # Optional: use LLM to get tags/ids for scoring (if API key present)
                if llm and getattr(llm, "api_key", ""):
                    try:
                        resp = llm.search(freeform)
                        if isinstance(resp, dict):
                            llm_tags = resp.get("tags", []) or []
                            llm_ids = resp.get("property_ids", []) or []
                    except Exception as e:
                        print("LLM search error:", e)

            # ---- Temporary user object with per-run inputs ----
            tmp_user = User(
                username=getattr(user, "username", ""),
                name=getattr(user, "name", ""),   # first name
                group_size=group_size,
                environment=env_in,
                budget_min=budget_min,
                budget_max=budget_max,
            )

            # ---- Rank ----
            top5 = rec.recommend(tmp_user, candidates, llm_tags=llm_tags, llm_ids=llm_ids)

            # ---- Always generate a blurb, even if freeform is empty; include first name ----
            blurb = None
            traveler_name = (getattr(user, "name", "") or "").strip()  # first name from profile
            if llm and getattr(llm, "api_key", ""):
                try:
                    if freeform:
                        # Augment freeform with the traveler name
                        blurb_prompt = f"{freeform}\n\nTraveler first name: {traveler_name or 'Guest'}."
                    else:
                        # Compose a concise prompt from structured answers + first name
                        parts = []
                        if loc:
                            parts.append(f"in or near '{loc}'")
                        if env_in:
                            parts.append(f"with a '{env_in}' vibe")
                        if group_size:
                            parts.append(f"for {group_size} guests")
                        if (budget_min or budget_max) and not (budget_min == 0 and budget_max == 0):
                            parts.append(f"around ${budget_min:.0f}-${budget_max:.0f}/night")

                        who = traveler_name or "Guest"
                        blurb_prompt = (
                                f"Write a short, upbeat travel blurb addressed to {who} about suitable vacation rentals "
                                + (", ".join(parts) + "." if parts else ".")
                        )

                    blurb_resp = llm.generate_travel_blurb(blurb_prompt)
                    if isinstance(blurb_resp, str) and not blurb_resp.startswith("ERROR:"):
                        blurb = blurb_resp
                    elif isinstance(blurb_resp, str) and blurb_resp.startswith("ERROR:"):
                        print("LLM blurb error:", blurb_resp)
                except Exception as e:
                    print("LLM blurb exception:", e)

            # ---- Export (CSV); use USERNAME to avoid blank-name files ----
            Path("output").mkdir(parents=True, exist_ok=True)
            fname = f"recommendations_{_safe_filename(user.username)}.csv"
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
            loc = input("Location contains (Click enter if none): ").strip().lower()
            if loc:
                d = d[d["location"].str.lower().str.contains(loc, na=False)]
            typ = input("Property type (or blank): ").strip().lower()
            if typ:
                d = d[d["property_type"].str.lower().str.contains(typ, na=False)]
            pmin = input("Minimum price per night: ").strip()
            pmax = input("Maximum price per night: ").strip()
            try:
                if pmin:
                    d = d[d["nightly_price"] >= float(pmin)]
                if pmax:
                    d = d[d["nightly_price"] <= float(pmax)]
            except Exception:
                pass
            gs = input("Group size: ").strip()
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
            print("Invalid choice, input a number")


if __name__ == "__main__":
    main()
