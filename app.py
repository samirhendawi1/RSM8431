# app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from user.UserManager import UserManager, User, _hash_pw, _validate_password  # uses your existing helpers
from properties.PropertyManager import PropertyManager, generate_properties_csv
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper
from smart_search import SmartSearch

# ---------------------- Config ----------------------
DATA_PATH = "data/property_final.csv"
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="StayFinder", page_icon="🏡", layout="wide")
st.title("🏡 StayFinder")

# ---------------------- Utilities ----------------------
def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")

def _compose_fallback_blurb(first_name, loc, env_in, group, bmin, bmax):
    who = first_name or "Guest"
    bits = []
    if loc: bits.append(f"in or near {loc}")
    if env_in: bits.append(f"with a {env_in} vibe")
    if group: bits.append(f"for {group} guests")
    if (bmin or bmax) and not (bmin == 0 and bmax == 0):
        bits.append(f"around ${bmin:.0f}-${bmax:.0f}/night")
    tail = (", ".join(bits) + ".") if bits else "."
    return f"Hi {who}, here are places that fit what you asked for {tail} I prioritized capacity, price fit, and your setting/type."

@st.cache_data
def load_properties(path):
    if not Path(path).exists():
        generate_properties_csv(path, 140)
    pm = PropertyManager(path)
    return pm, pm.properties

def _init_user_manager():
    if "um" not in st.session_state:
        st.session_state.um = UserManager()
    return st.session_state.um

def _get_api_key() -> str:
    """
    Priority:
      1) st.session_state.api_key  (from the API Key tab)
      2) st.secrets["OPENROUTER_API_KEY"] (if a secrets file exists)
      3) os.environ["OPENROUTER_API_KEY"]
    """
    key = (st.session_state.get("api_key") or "").strip()
    if key:
        return key
    try:
        val = st.secrets.get("OPENROUTER_API_KEY")
        if val:
            return str(val).strip()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()

def _llm_from_context():
    key = _get_api_key()
    if not key:
        return None
    os.environ["OPENROUTER_API_KEY"] = key
    try:
        return LLMHelper(csv_path=DATA_PATH, request_timeout=8)
    except Exception:
        return None

def _latest_recs_path_for_user(u: User) -> Path | None:
    """Prefer username-based CSV; fallback to name; then latest containing username."""
    by_user = OUTPUT_DIR / f"recommendations_{_safe_filename(u.username)}.csv"
    if by_user.is_file():
        return by_user
    by_name = OUTPUT_DIR / f"recommendations_{_safe_filename(u.name)}.csv" if u.name else None
    if by_name and by_name.is_file():
        return by_name
    try:
        outs = sorted(OUTPUT_DIR.glob("recommendations_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        for p in outs:
            if _safe_filename(u.username) in p.name:
                return p
    except Exception:
        pass
    return None

def _show_recs_preview(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        cols = [c for c in [
            "property_id","location","environment","property_type","nightly_price",
            "min_guests","max_guests","features","tags"
        ] if c in df.columns]
        st.caption(f"Latest recommendations file: {csv_path}")
        st.dataframe(df[cols].head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read {csv_path}: {e}")

# ---------------------- Load data & core helpers ----------------------
pm, df_all = load_properties(DATA_PATH)
rec = Recommender(top_k=5, weights=(0.4, 0.4, 0.2), llm_weight=0.25)
um: UserManager = _init_user_manager()

# ---------------------- Tabs (maps to CLI menu) ----------------------
tab_find, tab_browse, tab_filter, tab_api, tab_account = st.tabs([
    "🔎 Find Stays", "📚 Browse Properties", "🧭 Search & Filter", "🔐 API Key", "👤 Account"
])

# ---------------------- API Key Tab (OpenRouter) ----------------------
with tab_api:
    st.subheader("OpenRouter API Key")
    st.write("Provide your key to enable AI-generated blurbs. It’s stored **in session only** (not written to disk).")
    existing = st.session_state.get("api_key", "")
    key_input = st.text_input("API key", type="password", value=existing, placeholder="sk-or-v1_...", help="Session-only storage.")
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("Save key"):
            if key_input:
                st.session_state.api_key = key_input.strip()
                os.environ["OPENROUTER_API_KEY"] = st.session_state.api_key
                st.success("API key saved for this session.")
            else:
                st.session_state.pop("api_key", None)
                os.environ.pop("OPENROUTER_API_KEY", None)
                st.warning("Cleared API key.")
    with col2:
        if st.button("Test connection"):
            llm = _llm_from_context()
            if not llm or not getattr(llm, "api_key", ""):
                st.error("No valid key detected.")
            else:
                st.success("Key detected. LLM helper initialized.")
    key_now = _get_api_key()
    st.caption("Status")
    if key_now:
        st.success("Connected ✅")
    else:
        st.warning("No key set – using fallback blurbs.")


# ---------------------- Account Tab (1/2/3/4/5/9) ----------------------
with tab_account:
    st.subheader("Account")

    # SIGN IN (2) & SIGN UP (1)
    if not um.current_username:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Sign in")
            si_user = st.text_input("Username", key="si_user_tab")
            si_pw = st.text_input("Password", type="password", key="si_pw_tab")
            if st.button("Sign in", key="signin_btn_tab"):
                recd = um.users.get((si_user or "").strip().lower())
                if not recd:
                    st.error("No such user.")
                elif _hash_pw(si_pw, recd.get("salt","")) != recd.get("password_hash",""):
                    st.error("Invalid credentials.")
                else:
                    um.current_username = (si_user or "").strip().lower()
                    st.success(f"Signed in as {um.current_username}")
                    st.rerun()

        with c2:
            st.markdown("### Sign up")
            su_user = st.text_input("New username", key="su_user_tab")
            su_first = st.text_input("First name (optional)", key="su_first_tab")
            su_pw1 = st.text_input("Password", type="password", key="su_pw1_tab")
            su_pw2 = st.text_input("Confirm password", type="password", key="su_pw2_tab")
            if st.button("Create account", key="signup_btn_tab"):
                un = (su_user or "").strip().lower()
                if un in um.users:
                    st.error("That username already exists.")
                elif su_pw1 != su_pw2:
                    st.error("Passwords did not match.")
                else:
                    ok, msg = _validate_password(su_pw1, un)
                    if not ok:
                        st.error(f"Invalid password: {msg}")
                    else:
                        import secrets
                        salt = secrets.token_hex(16)
                        um.users[un] = {
                            "username": un,
                            "first_name": (su_first or "").strip(),
                            "salt": salt,
                            "password_hash": _hash_pw(su_pw1, salt),
                        }
                        um.current_username = un
                        um._save()
                        st.success(f"User '{un}' created and signed in.")
                        st.rerun()
    else:
        u = um.get_current_user()
        st.markdown("### View profile")
        st.write(f"**Username:** `{u.username}`")
        st.write(f"**First name:** `{u.name or '—'}`")

        # Latest recs preview (no scores)
        pick = _latest_recs_path_for_user(u)
        if pick:
            _show_recs_preview(pick)
        else:
            st.caption("No past recommendations found.")

        st.write("---")
        st.markdown("### Edit profile")
        with st.form("edit_profile_form"):
            new_first = st.text_input("First name", value=u.name or "")
            new_username = st.text_input("New username", value=u.username)
            change_pw = st.checkbox("Change password?")
            cur_pw = st.text_input("Current password", type="password") if change_pw else ""
            new_pw1 = st.text_input("New password", type="password") if change_pw else ""
            new_pw2 = st.text_input("Confirm new password", type="password") if change_pw else ""
            submitted = st.form_submit_button("Save changes")
        if submitted:
            recd = um.users.get(um.current_username)
            if not recd:
                st.error("User record not found.")
            else:
                # First name
                recd["first_name"] = (new_first or "").strip()
                # Username change
                new_un = (new_username or "").strip().lower()
                if new_un != um.current_username:
                    if new_un in um.users:
                        st.error("That username already exists.")
                    else:
                        um.users.pop(um.current_username, None)
                        recd["username"] = new_un
                        um.users[new_un] = recd
                        um.current_username = new_un
                # Password change
                if change_pw:
                    if _hash_pw(cur_pw, recd.get("salt","")) != recd.get("password_hash",""):
                        st.error("Current password is incorrect.")
                        st.stop()
                    ok, msg = _validate_password(new_pw1, um.current_username)
                    if not ok:
                        st.error(f"Invalid password: {msg}")
                        st.stop()
                    if new_pw1 != new_pw2:
                        st.error("Passwords did not match.")
                        st.stop()
                    import secrets
                    salt = secrets.token_hex(16)
                    recd["salt"] = salt
                    recd["password_hash"] = _hash_pw(new_pw1, salt)
                um._save()
                st.success("Profile updated.")
                st.rerun()

        st.write("---")
        st.markdown("### Sign out")
        if st.button("Sign out"):
            um.sign_out()
            st.success("Signed out.")
            st.rerun()

        st.write("---")
        st.markdown("### Delete profile")
        with st.expander("🛑 Danger zone", expanded=False):
            confirm = st.text_input("Type your username to confirm", key="del_confirm_tab")
            agree = st.checkbox("I understand this cannot be undone.", key="del_ack_tab")
            if st.button("Delete account", type="primary", key="del_btn_tab"):
                if not agree:
                    st.error("Please acknowledge the warning.")
                elif (confirm or "").strip().lower() != u.username:
                    st.error("Confirmation did not match your username.")
                else:
                    # Use dedicated method if you added one; else inline deletion
                    if hasattr(um, "delete_user_by_username"):
                        ok = um.delete_user_by_username(u.username)
                    else:
                        um.users.pop(u.username, None)
                        if um.current_username == u.username:
                            um.current_username = None
                        um._save()
                        ok = True
                    if ok:
                        st.success("Your account was deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete the account (user not found).")

# ---------------------- Browse Properties Tab (6) ----------------------
with tab_browse:
    st.subheader("Show properties")
    st.caption("Browse the full catalog. Use **Search & Filter** for targeted queries.")
    n = st.slider("Rows to preview", 50, 1000, 300, step=50)
    cols_default = [
        "property_id","location","environment","property_type","nightly_price",
        "min_guests","max_guests","features","tags"
    ]
    cols = [c for c in cols_default if c in df_all.columns]
    st.dataframe(df_all[cols].head(n), use_container_width=True)
    csv_bytes = df_all[cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download all (displayed columns)", csv_bytes, file_name="all_properties.csv", mime="text/csv")

# ---------------------- Search & Filter Tab (8) ----------------------
with tab_filter:
    st.subheader("Search and Filter Properties")
    d = df_all
    c1, c2, c3 = st.columns(3)
    loc = c1.text_input("Location").strip().lower()
    prop_type = c2.text_input("Property type").strip().lower()
    env_pref = c3.text_input("Environment contains (beach/city/mountain/desert/lake)").strip().lower()

    c4, c5, c6 = st.columns(3)
    pmin = c4.number_input("Min price", min_value=0, value=0, step=10)
    pmax = c5.number_input("Max price", min_value=0, value=0, step=10)
    gs = c6.number_input("Group size", min_value=0, value=0, step=1)

    if loc:
        d = d[d["location"].str.lower().str.contains(loc, na=False)]
    if prop_type:
        d = d[d["property_type"].str.lower().str.contains(prop_type, na=False)]
    if env_pref:
        d = d[d["environment"].str.lower().str.contains(env_pref, na=False)]
    if pmin:
        d = d[d["nightly_price"] >= float(pmin)]
    if pmax:
        d = d[d["nightly_price"] <= float(pmax)]
    if gs:
        d = d[(d["min_guests"] <= int(gs)) & (int(gs) <= d["max_guests"])]

    cols = [c for c in [
        "property_id","location","environment","property_type","nightly_price",
        "min_guests","max_guests","features","tags"
    ] if c in d.columns]
    st.write(f"Results: **{len(d)}**")
    st.dataframe(d[cols].head(1000), use_container_width=True)
    csv_bytes = d[cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", csv_bytes, file_name="filtered_properties.csv", mime="text/csv")

# ---------------------- Find Stays Tab (7) ----------------------
with tab_find:
    st.subheader("Get recommendations (exports CSV)")

    if not um.current_username:
        st.info("Sign in on the **Account** tab to get recommendations.")
        st.stop()

    u = um.get_current_user()
    c1, c2, c3, c4, c5 = st.columns(5)
    loc = c1.text_input("Location contains (optional)")
    env_in = c2.text_input("Environment (e.g., beach, city, mountain)")
    group_size = c3.number_input("Group size", min_value=0, step=1, value=0)
    bmin = c4.number_input("Budget min", min_value=0, step=10, value=0)
    bmax = c5.number_input("Budget max", min_value=0, step=10, value=0)
    freeform = st.text_area("Extra details (optional)")

    if st.button("Get recommendations", key="go_btn_find"):
        # Filter by location pre-step
        candidates = df_all
        if loc:
            try:
                candidates = candidates[candidates["location"].str.lower().str.contains(loc.lower(), na=False)]
            except Exception:
                pass

        # Smart search & optional LLM tags
        llm_tags, llm_ids = [], []
        if freeform:
            try:
                ss = SmartSearch(candidates)
                candidates = ss.find_candidates(freeform, top_k=300)
            except Exception:
                pass
            llm = _llm_from_context()
            if llm and getattr(llm, "api_key", ""):
                try:
                    resp = llm.search(freeform)
                    if isinstance(resp, dict):
                        llm_tags = resp.get("tags", []) or []
                        llm_ids = resp.get("property_ids", []) or []
                except Exception:
                    pass

        # Temporary User object (per-run only)
        tmp_user = User(
            username=u.username,
            name=u.name,
            group_size=int(group_size or 0),
            environment=(env_in or ""),
            budget_min=float(bmin or 0),
            budget_max=float(bmax or 0),
        )

        # Rank
        top5 = rec.recommend(tmp_user, candidates, llm_tags=llm_tags, llm_ids=llm_ids)

        # Blurb (LLM or fallback), always include first name
        blurb = None
        llm = _llm_from_context()
        if llm and getattr(llm, "api_key", ""):
            try:
                if freeform:
                    prompt = f"{freeform}\n\nTraveler first name: {u.name or 'Guest'}."
                else:
                    parts = []
                    if loc: parts.append(f"in or near '{loc}'")
                    if env_in: parts.append(f"with a '{env_in}' vibe")
                    if group_size: parts.append(f"for {int(group_size)} guests")
                    if (bmin or bmax) and not (bmin == 0 and bmax == 0):
                        parts.append(f"around ${bmin:.0f}-${bmax:.0f}/night")
                    who = u.name or "Guest"
                    prompt = (
                            f"Write a short, upbeat travel blurb addressed to {who} about suitable vacation rentals "
                            + (", ".join(parts) + "." if parts else ".")
                    )
                resp = llm.generate_travel_blurb(prompt)
                if isinstance(resp, str) and not resp.startswith("ERROR:"):
                    blurb = resp
            except Exception:
                pass
        if not blurb:
            blurb = _compose_fallback_blurb(u.name, loc, env_in, int(group_size or 0), float(bmin or 0), float(bmax or 0))

        st.markdown("### ✍️ Blurb")
        st.write(blurb)

        # Display without score columns
        cols = [c for c in [
            "property_id","location","environment","property_type","nightly_price",
            "min_guests","max_guests","features","tags"
        ] if c in top5.columns]
        st.markdown("### 🔎 Top matches")
        st.dataframe(top5[cols], use_container_width=True)

        # Export with username-based filename (avoid blank-name files)
        fname = f"recommendations_{_safe_filename(u.username)}.csv"
        csv_bytes = top5.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")
