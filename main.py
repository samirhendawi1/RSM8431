import os
from pathlib import Path
import pandas as pd

from user.UserManager import UserManager
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper
from smart_search import SmartSearch

DATA_CSV = "data/property_final.csv"  # read-only source


def _safe_filename(s: str) -> str:
    s = s or ""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")


def _load_properties_readonly(path: str) -> pd.DataFrame:
    """Strictly read the CSV. Never writes, never generates."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CSV at {path}. Add it to /data and retry.")
    # Read + light normalization in-memory only
    df = pd.read_csv(path)
    # Normalize common columns without mutating disk
    df.columns = [str(c).strip() for c in df.columns]
    # Ensure expected columns exist (in-memory only)
    for c in ["property_id","location","environment","property_type",
              "nightly_price","features","tags","min_guests","max_guests"]:
        if c not in df.columns:
            df[c] = 0 if c in ("nightly_price","min_guests","max_guests","property_id") else ""
    # Types
    df["nightly_price"] = pd.to_numeric(df["nightly_price"], errors="coerce").fillna(0.0)
    for c in ["min_guests","max_guests","property_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["features","tags","location","environment","property_type"]:
        df[c] = df[c].fillna("").astype(str)
    # Fulltext (for SmartSearch/Jaccard)
    df["__fulltext__"] = (
            df["location"] + " " + df["environment"] + " " + df["property_type"] + " " + df["features"] + " " + df["tags"]
    ).str.lower()
    return df


def main():
    # ---------- Data (read-only) ----------
    properties = _load_properties_readonly(DATA_CSV)

    # ---------- Core managers ----------
    user_manager = UserManager()
    rec = Recommender(top_k=5, weights=(0.4, 0.4, 0.2), llm_weight=0.25)
    ss = SmartSearch(properties)

    # LLM helper (reads CSV internally for prompts; read-only)
    try:
        llm = LLMHelper(csv_path=DATA_CSV, request_timeout=8)
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
            # Username only, then last 5 recs (as you asked previously)
            user_manager.view_profile()
            u = user_manager.get_current_user()
            if not u:
                continue

            # Prefer username-based file; fallback to name; then latest containing username
            username_candidate = Path("output") / f"recommendations_{_safe_filename(u.username)}.csv"
            name_candidate = Path("output") / f"recommendations_{_safe_filename(u.name)}.csv" if getattr(u, "name", None) else None

            pick = None
            if username_candidate.is_file():
                pick = username_candidate
            elif name_candidate and name_candidate.is_file():
                pick = name_candidate
            else:
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
                    df = pd.read_csv(pick)
                    print(f"Latest recommendations file: {pick}")
                    show_cols = [c for c in [
                        "property_id","location","environment","property_type","nightly_price",
                        "min_guests","max_guests","features","tags"
                    ] if c in df.columns]
                    print(df[show_cols].head(5).to_string(index=False))
                except Exception as e:
                    print(f"Could not read {pick}: {e}")

        elif choice == "6":
            # Display from in-memory DataFrame only
            cols = ["property_id","location","environment","property_type","nightly_price",
                    "min_guests","max_guests","features","tags"]
            cols = [c for c in cols if c in properties.columns]
            print("\nAvailable Properties:")
            print(properties[cols].head(50).to_string(index=False))

        elif choice == "7":
            user = user_manager.get_current_user()
            if not user:
                print("Sign in first.")
                continue

            # Collect per-run preferences (session-only; NOT saved)
            print("\n-- Recommendation Inputs (blank keeps default) --")
            loc = input("Location contains (optional): ").strip()
            env_in = input(f"Environment [{user.environment or 'none'}]: ").strip().lower() or user.environment
            gs_in = input(f"Group size [{user.group_size}]: ").strip()
            bmin_in = input(f"Budget min [{user.budget_min}]: ").strip()
            bmax_in = input(f"Budget max [{user.budget_max}]: ").strip()
            freeform = input("Extra details (optional free-text): ").strip()

            # Safe parsing with fallbacks
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

            # Build candidates from in-memory DataFrame
            candidates = properties
            if loc:
                try:
                    candidates = candidates[candidates["location"].str.lower().str.contains(loc.lower(), na=False)]
                except Exception:
                    pass

            # SmartSearch + optional LLM tags/ids
            llm_tags, llm_ids = [], []
            if freeform:
                try:
                    candidates = SmartSearch(candidates).find_candidates(freeform, top_k=300)
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

            # Rank
            tmp_user = type("UserLite",(object,),{})()
            tmp_user.username = getattr(user, "username", "")
            tmp_user.name = getattr(user, "name", "")
            tmp_user.group_size = group_size
            tmp_user.environment = env_in or ""
            tmp_user.budget_min = budget_min
            tmp_user.budget_max = budget_max

            top5 = rec.recommend(tmp_user, candidates, llm_tags=llm_tags, llm_ids=llm_ids)

            # Blurb (LLM or fallback), always include first name
            blurb = None
            if llm and getattr(llm, "api_key", ""):
                try:
                    if freeform:
                        blurb_prompt = f"{freeform}\n\nTraveler first name: {tmp_user.name or 'Guest'}."
                    else:
                        parts = []
                        if loc: parts.append(f"in or near '{loc}'")
                        if env_in: parts.append(f"with a '{env_in}' vibe")
                        if group_size: parts.append(f"for {group_size} guests")
                        if (budget_min or budget_max) and not (budget_min == 0 and budget_max == 0):
                            parts.append(f"around ${budget_min:.0f}-${budget_max:.0f}/night")
                        who = tmp_user.name or "Guest"
                        blurb_prompt = (
                                f"Write a short, upbeat travel blurb addressed to {who} about suitable vacation rentals "
                                + (', '.join(parts) + '.' if parts else '.')
                        )
                    resp = llm.generate_travel_blurb(blurb_prompt)
                    if isinstance(resp, str) and not resp.startswith("ERROR:"):
                        blurb = resp
                    elif isinstance(resp, str) and resp.startswith("ERROR:"):
                        print("LLM blurb error:", resp)
                except Exception as e:
                    print("LLM blurb exception:", e)

            # Export (username-based filename; read-only dataset remains untouched)
            Path("output").mkdir(parents=True, exist_ok=True)
            out = Path("output") / f"recommendations_{_safe_filename(user.username)}.csv"
            try:
                top5.to_csv(out, index=False, encoding="utf-8")
            except Exception as e:
                print(f"Failed to save CSV: {e}")

            if blurb:
                print("\n— Blurb —")
                print(blurb)

            # Display without internal score columns
            show_cols = [c for c in [
                "property_id","location","environment","property_type","nightly_price",
                "min_guests","max_guests","features","tags"
            ] if c in top5.columns]
            try:
                print(f"\nTop 5 saved to: {out}")
                print(top5[show_cols].to_string(index=False))
            except Exception:
                print(f"\nTop 5 saved to: {out}")
                print(top5[show_cols].head(5))

        elif choice == "8":
            d = properties
            print(f"\nRows in dataset: {len(d)}")
            loc = input("Location contains (or blank): ").strip().lower()
            if loc:
                d = d[d["location"].str.lower().str.contains(loc, na=False)]
            typ = input("Property type (or blank): ").strip().lower()
            if typ:
                d = d[d["property_type"].str.lower().str.contains(typ, na=False)]
            pmin = input("Minimum price per night: ").strip()
            pmax = input("Maximum price per night: ").strip()
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
            cols = [c for c in [
                "property_id","location","environment","property_type","nightly_price",
                "min_guests","max_guests","features","tags"
            ] if c in d.columns]
            print(d[cols].head(50).to_string(index=False))

        elif choice == "9":
            user_manager.delete_user()

        elif choice == "0":
            print("Bye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
