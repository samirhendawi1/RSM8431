# RSM8431 — Property Recommender (CLI)

A simple command-line app that loads a small properties dataset and recommends stays based on your profile (environment + budget). It also supports a lightweight smart search and optional LLM-assisted tagging.

---

## Features

* **User profiles**: name, group size, preferred environment, budget range, travel dates
* **Recommendations**: returns Top-5 with a composite `fit_score` and saves them to CSV
* **Smart Search**: free-text search over features/tags (e.g., `quiet beach surfing hot tub`)
* **CSV data**: ships with sample data in `data/`
* **Optional LLM boost**: improves ranking when `OPENROUTER_API_KEY` is set

---

## Project Structure

```
.
├─ main.py                      # CLI entrypoint
├─ smart_search.py              # Free-text candidate finder / tag canonicalizer
├─ recommender/
│  ├─ Recommender.py            # Scoring & Top-K selection
│  └─ llm.py                    # (Optional) LLM helper (OpenRouter)
├─ properties/
│  └─ PropertyManager.py        # CSV loader + (utility) CSV generator
├─ user/
│  └─ UserManager.py            # User and UserManager classes
├─ data/
│  ├─ properties_expanded.csv   # Default dataset (preferred)
│  └─ properties.csv            # Fallback dataset
└─ output/
   └─ recommendations_*.csv     # Generated results
```

---

## Requirements

* Python **3.9+**
* Python packages: `pandas`

Install deps:

```bash
pip install pandas
```

> Tip: Use a venv.
>
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate          # Windows: .venv\Scripts\activate
> pip install pandas
> ```

---

## Quickstart

From the project root (where `main.py` lives):

```bash
python main.py
```

You’ll see a menu like:

```
Menu:
1. Create user
2. Edit profile
3. View profile
4. Show properties
5. Get recommendations
6. Smart search
7. Delete profile
8. Exit
```

### Minimal flow (example)

1 → Create user (enter: ID, Name, Group size, Environment like `beach`/`mountain`/`city`/`lake`, Budget min/max, Travel dates)
4 → Show properties (preview the dataset)
5 → Get recommendations (optionally type a free-form query; press Enter to skip)
8 → Exit

Results are saved to:

```
output/recommendations_<YourName>.csv
```

---

## Data Schema

`data/properties_expanded.csv` (or `properties.csv`) columns:

* `location` – city/region string
* `type` – property/environment type (e.g., beach, mountain, city, lake)
* `nightly_price` – integer price per night
* `features` – comma-separated list (e.g., `WiFi, hot tub`)
* `tags` – comma-separated list (e.g., `family-friendly, remote`)

**Recommendation Output** adds:

* `fit_score` – composite score combining environment match, budget fit, and search/semantic boosts (higher is better)

---

## How scoring works (high level)

* **Environment match**: preferred types (e.g., `beach`) are prioritized
* **Budget fit**: properties within `[budget_min, budget_max]` get higher weight
* **Search boosts**: matches from Smart Search and (optionally) LLM tags can nudge items up

Exact weights are in `recommender/Recommender.py`.

---

## Smart Search

Menu → **6. Smart search**
Enter free text (keywords, features, tags). Example:

```
quiet beach surfing hot tub
```

Returns the most relevant candidates (top 10).

---

## LLM-assisted ranking

If you want to enhance results with LLM-derived tags/IDs:

1. Get an OpenRouter API key
2. Export it before running:

```bash
export OPENROUTER_API_KEY="sk-..."
# Windows (PowerShell): $env:OPENROUTER_API_KEY="sk-..."
```

The recommender will use it (when free-form text is provided) to slightly boost candidates that align with LLM-suggested tags/IDs.

---

## Generate a fresh dataset (optional)

`properties/PropertyManager.py` includes a helper to generate a CSV.

Example script:

```python
from properties.PropertyManager import generate_properties_csv
generate_properties_csv(filename="data/properties.csv", count=100)
```

---

## Troubleshooting

* **`ModuleNotFoundError` or imports failing:** run `python main.py` **from the project root** so relative imports and `data/` are resolved correctly.
* **No output CSV:** ensure you ran **5. Get recommendations**, budgets are valid (min ≤ max, both > 0), and the `output/` folder is writable.
* **No properties shown:** verify `data/properties_expanded.csv` or `data/properties.csv` exists and has the expected columns.

---

## Short Test Plan

1. Launch: `python main.py`
2. Create user:

   * Environment: `beach`
   * Budget: `120` to `250`
3. Show properties (menu 4) – should print a table.
4. Get recommendations (menu 5) – confirm it prints a path like:

   ```
   Top 5 recommendations saved to: output/recommendations_<YourName>.csv
   ```
5. Open the CSV and verify columns include `location`, `type`, `nightly_price`, `features`, `fit_score`.

---

## Notes

* The CLI prefers `data/properties_expanded.csv`; if missing, it falls back to `data/properties.csv`.
* You can freely edit/extend the scoring logic or add new filters (e.g., max nightly price, tag filters) in the recommender and CLI.
