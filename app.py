# app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from user.UserManager import UserManager, User
from properties.PropertyManager import PropertyManager, generate_properties_csv
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper
from smart_search import SmartSearch

# ---------------------- Config ----------------------
DATA_PATH = "data/properties_with_capacity_types.csv"
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
    # 1) session
    key = (st.session_state.get("api_key") or "").strip()
    if key:
        return key

    # 2) secrets (guard against missing file)
    try:
        val = st.secrets.get("OPENROUTER_API_KEY")  # may raise if no secrets.toml
        if val:
            return str(val).strip()
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # 3) env var
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()

def _llm_from_context():
    key = _get_api_key()
    if not key:
        return None
    # Ensure LLMHelper sees it
    os.environ["OPENROUTER_API_KEY"] = key
    try:
        return LLMHelper(csv_path=DATA_PATH, request_timeout=8)
    except Exception:
        return None

# ---------------------- Load data & core helpers ----------------------
pm, df_all = load_properties(DATA_PATH)
rec = Recommender(top_k=5, weights=(0.4, 0.4, 0.2), llm_weight=0.25)
um: UserManager = _init_user_manager()

# ---------------------- Tabs ----------------------
tab_find, tab_api, tab_account = st.tabs(["🔎 Find Stays", "🔐 API Key", "👤 Account"])

# ---------------------- API Key Tab ----------------------
with tab_api:
    st.subheader("OpenRouter API Key")
    st.write("Provide your key to enable AI-generated blurbs. It’s stored **in session only** (not written to disk).")

    existing = st.session_state.get("api_key", "")
    key_input = st.text_input("API key", type="password", value=existing, placeholder="sk-or-v1_...", help="Session-only storage.")

    col1, col2, col3 = st.columns([1, 1, 2])
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

    # Status pill
    key_now = _get_api_key()
    if key_now:
        st.markdown("**Status:** ✅ Using API key from session/secrets/env.")
    else:
        st.markdown("**Status:** ⚠️ No API key set. Fallback blurbs will be used.")

# ---------------------- Account Tab ----------------------
with tab_account:
    st.subheader("Account")

    if not um.current_username:
        st.write("### Sign in")
        si_user = st.text_input("Username", key="si_user_tab")
        si_pw = st.text_input("Password", type="password", key="si_pw_tab")
        if st.button("Sign in", key="signin_btn_tab"):
            recd = um.users.get((si_user or "").strip().lower())
            if not recd:
                st.error("No such user.")
            else:
                from user.UserManager import _hash_pw
                if _hash_pw(si_pw, recd.get("salt", "")) == recd.get("password_hash", ""):
                    um.current_username = (si_user or "").strip().lower()
                    st.success(f"Signed in as {um.current_username}")
                else:
                    st.error("Invalid credentials.")

        st.write("---")
        st.write("### Sign up")
        su_user = st.text_input("New username", key="su_user_tab")
        su_first = st.text_input("First name (optional)", key="su_first_tab")
        su_pw1 = st.text_input("Password", type="password", key="su_pw1_tab")
        su_pw2 = st.text_input("Confirm password", type="password", key="su_pw2_tab")
        if st.button("Create account", key="signup_btn_tab"):
            if (su_user or "").strip().lower() in um.users:
                st.error("That username already exists.")
            elif su_pw1 != su_pw2:
                st.error("Passwords did not match.")
            else:
                from user.UserManager import _hash_pw, _validate_password
                ok, msg = _validate_password(su_pw1, (su_user or "").strip().lower())
                if not ok:
                    st.error(f"Invalid password: {msg}")
                else:
                    import secrets
                    salt = secrets.token_hex(16)
                    um.users[(su_user or "").strip().lower()] = {
                        "username": (su_user or "").strip().lower(),
                        "first_name": (su_first or "").strip(),
                        "salt": salt,
                        "password_hash": _hash_pw(su_pw1, salt),
                    }
                    um.current_username = (su_user or "").strip().lower()
                    um._save()
                    st.success(f"User '{um.current_username}' created and signed in.")
    else:
        u = um.get_current_user()
        st.write(f"**Signed in as:** `{u.username}`")
        st.write(f"**First name:** `{u.name or '—'}`")
        if st.button("Sign out (this tab)"):
            um.sign_out()
            st.success("Signed out.")

# ---------------------- Find Stays Tab ----------------------
with tab_find:
    st.subheader("Your trip")

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
        # Filter by location
        candidates = df_all
        if loc:
            try:
                candidates = candidates[candidates["location"].str.lower().str.contains(loc.lower(), na=False)]
            except Exception:
                pass

        # Smart search for freeform
        llm_tags, llm_ids = [], []
        if freeform:
            try:
                ss = SmartSearch(candidates)
                candidates = ss.find_candidates(freeform, top_k=300)
            except Exception:
                pass
            # Optional tags/ids via LLM
            llm = _llm_from_context()
            if llm and getattr(llm, "api_key", ""):
                try:
                    resp = llm.search(freeform)
                    if isinstance(resp, dict):
                        llm_tags = resp.get("tags", []) or []
                        llm_ids = resp.get("property_ids", []) or []
                except Exception:
                    pass

        # Temporary User object with per-run inputs (not persisted)
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

        # Blurb (always attempt LLM; fallback otherwise)
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
            "property_id", "location", "environment", "property_type", "nightly_price",
            "min_guests", "max_guests", "features", "tags"
        ] if c in top5.columns]
        st.markdown("### 🔎 Top matches")
        st.dataframe(top5[cols], use_container_width=True)

        # Download CSV
        fname = f"recommendations_{_safe_filename(u.username)}.csv"
        csv_bytes = top5.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")
