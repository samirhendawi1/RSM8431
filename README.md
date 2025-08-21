# RSM8431 — Property Recommender

A lightweight property recommendation system with a **CLI** and an optional **Streamlit UI**. It loads a sample dataset, lets users create simple profiles, runs a rules-plus-signals recommender (with optional LLM hints), and exports personalized results to CSV.

---

## Key Features

* **User profiles**: username, optional first name, group size, preferred environment, budget range
* **Top-K recommendations**: ranked by a composite `fit_score` (environment, budget fit, capacity, search/semantic boosts)
* **Smart Search**: free-text filter over features/tags/locations with simple synonym handling
* **Optional LLM boost**: structured hints (tags/features/locations/environments) via OpenRouter (if key is set)
* **CSV outputs**: saved under `output/` per user

---

## Project Structure

```text
.
├─ app.py                       # Streamlit UI (optional)
├─ main.py                      # CLI entrypoint (recommended for quick use)
├─ smart_search.py              # Free-text candidate finder / tag canonicalizer
├─ recommender/
│  ├─ Recommender.py            # Scoring & Top-K selection
│  ├─ llm.py                    # Stable import shim for LLM helper
│  └─ LLMHelper.py              # Optional: OpenRouter client for search hints
├─ properties/
│  └─ PropertyManager.py        # CSV loader + utilities
├─ user/
│  └─ UserManager.py            # Demo user store (salted password hashes)
├─ data/
│  ├─ property_final.csv        # Property dataset (read-only source)
│  └─ users.csv                 # Demo users (do not use for real credentials)
├─ output/
│  └─ recommendations_*.csv     # Generated examples + your runs
├─ secrets.toml                 # (Optional) Streamlit secrets
└─ APIKEY                       # (Legacy/unused) — safe to remove
```

> Note: Your archive may include `__MACOSX/`, `__pycache__/`, `.idea/`, `.DS_Store`, and some pre-generated `output/*.csv`. They’re not required for running.

---

## Requirements

* **Python** 3.9+
* **CLI**: `pandas`
* **Streamlit UI (optional)**: `streamlit`
* **LLM hints (optional)**: `requests` and an `OPENROUTER_API_KEY`

Install the minimal set you need, e.g.:

```bash
# CLI only
pip install pandas

# CLI + Streamlit UI
pip install pandas streamlit

# Add LLM hints support
pip install requests
```

---

## Data Schema

**Source:** `data/property_final.csv`

| column          | type | example / notes                                                  |
| --------------- | ---- | ---------------------------------------------------------------- |
| `property_id`   | str  | `P000123`                                                        |
| `location`      | str  | `Phuket, Thailand`                                               |
| `environment`   | str  | e.g., `beach`, `mountain`, `city`, `lake`, `island`, `forest`, … |
| `property_type` | str  | e.g., `apartment`, `villa`, `guesthouse`, `cabin`, …             |
| `nightly_price` | int  | price per night                                                  |
| `features`      | str  | comma-separated (e.g., `WiFi, hot tub, kitchen`)                 |
| `tags`          | str  | comma-separated (e.g., `family-friendly, long stay, beachfront`) |
| `min_guests`    | int  | minimum capacity                                                 |
| `max_guests`    | int  | maximum capacity                                                 |

**Outputs** (under `output/`) add at least:

* `fit_score` – composite score; higher is better

---

## Quick Start

### Run the CLI

```bash
python main.py
```

You’ll get a simple menu to sign up/sign in, set profile details, show properties, get recommendations (exports CSV), or search/filter.

**Where results go**
`output/recommendations_<username>.csv`

### Run the Streamlit UI (optional)

```bash
streamlit run app.py
```

* Set your profile in the sidebar
* Use the search box for free-form filters (e.g., `quiet beach hot tub`)
* Download recommendations from the page

---

## Smart Search (free-text)

* Enter space-separated cues: `beach quiet wifi hot tub phuket`
* Built-in synonyms include a few helpful normalizations (e.g., `hottub` → `hot tub`, `wi-fi` → `wifi`, `downtown` → `city`).
* Search runs over a composed field from `location`, `environment`, `property_type`, `features`, and `tags`.

---

## Optional: LLM Hints (OpenRouter)

If `OPENROUTER_API_KEY` is present, the app calls a small helper that extracts structured hints (`tags`, `features`, `locations`, `environments`, `property_ids`) from your free text to slightly boost relevant items.

**Set the key (choose one):**

```bash
# Environment variable (works for CLI and Streamlit)
export OPENROUTER_API_KEY="sk-..."

# OR Streamlit secrets (add to .streamlit/secrets.toml or project secrets.toml)
[general]
OPENROUTER_API_KEY = "sk-..."
```

> If the key is missing or requests fail, the recommender still works, just without LLM boosts.

---

## How Scoring Works (high level)

* **Environment match**: favor properties whose `environment` matches your preference
* **Budget fit**: closeness to your `[budget_min, budget_max]` range
* **Group capacity**: compatibility with `min_guests` / `max_guests`
* **Search/semantic boosts**: matches from Smart Search and (optionally) LLM hints
* Final `fit_score` is a blend of these components (see `recommender/Recommender.py`)

---

## Security & Data Notes

* `data/users.csv` stores **salted password hashes** for demo purposes only. Do **not** use real credentials.
* Treat `data/property_final.csv` as **read-only**. If you need to modify data, write to a copy.

---

## Housekeeping (.gitignore suggestion)

```gitignore
# OS/IDE
.DS_Store
__MACOSX/
.idea/

# Python
__pycache__/
*.pyc

# App outputs
output/*.csv

# Local configs (if any)
secrets.toml
.streamlit/
```

---

## Troubleshooting

* **Module not found (`pandas`/`streamlit`/`requests`)**
  → Install the listed dependencies for your chosen mode.
* **No CSV output appears**
  → Ensure `output/` exists (the app tries to create it) and you completed the menu flow.
* **LLM hints not applied**
  → Verify `OPENROUTER_API_KEY` is set. The app falls back gracefully if not.
* **Dataset path errors**
  → `main.py`/`app.py` expect `data/property_final.csv` relative to the project root.

---

## License

Academic/demo use
