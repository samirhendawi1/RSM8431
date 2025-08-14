from __future__ import annotations
import os
import json
import getpass
import requests
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("Pandas is required for LLMHelper (pip install pandas).") from e


DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


SYSTEM_PROMPT_JSON = (
    "You are an assistant for an Airbnb-like vacation property search. "
    "Given a list of properties (JSON) and a user request, return either: "
    "(1) a JSON object with `tags` (list of strings) inferred from the request, and "
    "(2) optionally `property_ids` (list of integers) that might match. "
    "Only return valid JSON. Do not include commentary."
)

SYSTEM_PROMPT_BLURB = (
    "You write short, lively travel blurbs for vacation rentals. "
    "Keep it under 80 words. No emojis, no markdown."
)


class LLMHelper:
    """
    - Loads properties from a CSV (adds a numeric `property_id` if missing).
    - Provides:
        * search(user_prompt) -> dict with keys: tags [list[str]], property_ids [list[int]] (optional)
        * generate_travel_blurb(prompt) -> str
    - API key resolution order:
        1) explicit api_key argument
        2) env var OPENROUTER_API_KEY
        3) interactive prompt via getpass (hidden input)
    """

    def __init__(
            self,
            csv_path: Optional[str] = None,
            api_key: Optional[str] = None,
            model: str = DEFAULT_MODEL,
            request_timeout: int = 60,
    ) -> None:
        self.model = model
        self.timeout = max(1, int(request_timeout)) if isinstance(request_timeout, int) else 60

        # Resolve API key; tolerate environments where input() isn't possible
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_key:
            try:
                resolved_key = self._prompt_api_key().strip()
            except Exception:
                resolved_key = ""

        self.api_key = resolved_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        } if self.api_key else None

        # Load properties (as list of dicts)
        self.properties = self._load_properties_csv(csv_path)

    # -------------------------------
    # Public API
    # -------------------------------
    def search(self, user_prompt: str) -> Dict[str, Any]:
        """
        Call the LLM and return a parsed dict.
        On error, returns {'error': '...', 'details'?: ... , 'raw'?: ...}
        """
        if not isinstance(user_prompt, str):
            user_prompt = "" if user_prompt is None else str(user_prompt)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {
                    "role": "user",
                    "content": (
                        "PROPERTIES:\n" + self._safe_dump_json(self.properties) +
                        "\n\nUSER REQUEST:\n" + user_prompt +
                        '\n\nRespond with JSON: {"tags": [...], "property_ids": [...]} (property_ids optional)'
                    ),
                },
            ],
            "temperature": 0.2,
        }
        return self._post_and_parse(payload)

    def generate_travel_blurb(self, prompt: str) -> str:
        """
        Returns a short descriptive blurb string.
        On failure, returns an error string prefixed with 'ERROR:'.
        """
        if not isinstance(prompt, str):
            prompt = "" if prompt is None else str(prompt)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_BLURB},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
        }
        parsed = self._post_and_parse(payload, expect_json=False)
        if isinstance(parsed, dict) and "error" in parsed:
            # Bubble up a readable message
            msg = parsed.get("error", "Unknown error")
            details = parsed.get("details")
            if isinstance(details, str):
                return f"ERROR: {msg} | {details[:240]}"
            return f"ERROR: {msg}"
        return parsed if isinstance(parsed, str) else str(parsed)

    # -------------------------------
    # Internal helpers
    # -------------------------------
    def _post_and_parse(self, payload: Dict[str, Any], expect_json: bool = True) -> Any:
        # Pre-flight checks
        if not isinstance(payload, dict):
            return {"error": "Invalid request payload", "details": "payload must be a dict"}

        if not self.api_key or not self.headers:
            return {"error": "Missing API key", "details": "Set OPENROUTER_API_KEY or pass api_key to LLMHelper."}

        try:
            r = requests.post(OPENROUTER_URL, headers=self.headers, json=payload, timeout=self.timeout)
        except Exception as e:
            return {"error": "Request failed", "details": str(e)}

        if not getattr(r, "status_code", None):
            return {"error": "No HTTP response", "details": "requests returned no status code"}

        if r.status_code != 200:
            # Try to capture JSON body if present; else raw text
            try:
                body = r.json()
            except Exception:
                body = r.text[:1000] if hasattr(r, "text") else ""
            return {"error": f"HTTP {r.status_code}", "details": body}

        # Parse OpenRouter JSON
        try:
            data = r.json()
        except Exception as e:
            return {"error": "Invalid JSON from API", "details": str(e)}

        # Extract model message content
        try:
            choices = data.get("choices") or []
            if not choices:
                return {"error": "Empty response", "details": data}
            msg = choices[0].get("message", {}).get("content")
        except Exception:
            msg = None

        if not msg:
            return {"error": "No content in response", "details": data}

        if not expect_json:
            # Return raw assistant text for blurb generation
            return msg

        # Try strict JSON parse first
        try:
            return json.loads(msg)
        except json.JSONDecodeError:
            # Loose extraction if model included extra text
            start = msg.find("{")
            end = msg.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(msg[start:end + 1])
                except json.JSONDecodeError:
                    return {"error": "Model returned non-JSON content", "raw": msg[:1000]}
            return {"error": "Model returned non-JSON content", "raw": msg[:1000]}
        except Exception as e:
            return {"error": "JSON parse error", "details": str(e)}

    def _prompt_api_key(self) -> str:
        try:
            return getpass.getpass("Enter your OpenRouter API key: ").strip()
        except Exception:
            try:
                return input("Enter your OpenRouter API key: ").strip()
            except Exception:
                return ""

    def _load_properties_csv(self, csv_path: Optional[str]) -> List[Dict[str, Any]]:
        # Resolve path with a few sensible fallbacks
        candidate_paths = [
            csv_path,
            "data/properties.csv",
            "properties.csv",
            "properties_expanded.csv",
        ]
        candidate_paths = [p for p in candidate_paths if p]

        csv_file = None
        for p in candidate_paths:
            try:
                if os.path.isfile(p):
                    csv_file = p
                    break
            except Exception:
                continue

        if csv_file is None:
            # Return empty set instead of raising; caller can still use search (will just analyze tags)
            return []

        # Read CSV defensively
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            # On read failure, return empty property list rather than crash
            return []

        # Normalize column names
        try:
            df.columns = [str(c).strip().lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame(df)
            df.columns = [str(c).strip().lower() for c in df.columns]

        # Ensure there's a stable integer `property_id`
        if "property_id" not in df.columns:
            try:
                df = df.reset_index().rename(columns={"index": "property_id"})
                df["property_id"] = pd.to_numeric(df["property_id"], errors="coerce").fillna(0).astype(int)
            except Exception:
                # Worst case: synthesize an id column
                df = df.copy()
                df["property_id"] = range(len(df))

        # Basic type cleanup
        if "nightly_price" in df.columns:
            try:
                df["nightly_price"] = pd.to_numeric(df["nightly_price"], errors="coerce")
            except Exception:
                pass

        # Convert NaN to empty strings for JSON friendliness
        try:
            records = df.fillna("").to_dict(orient="records")
        except Exception:
            # As a last resort, coerce row by row
            records = []
            try:
                for _, row in df.iterrows():
                    try:
                        rec = {str(k): ("" if (v is None) else (str(v) if not isinstance(v, (int, float, bool, dict, list)) else v))
                               for k, v in row.items()}
                        # Keep numerics as-is when possible
                        if "property_id" in row:
                            rec["property_id"] = int(row["property_id"]) if pd.notna(row["property_id"]) else 0
                        records.append(rec)
                    except Exception:
                        continue
            except Exception:
                records = []

        return records

    def _safe_dump_json(self, obj: Any) -> str:
        """Dump JSON safely for prompt construction (never raises)."""
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            try:
                # Best-effort: convert to str if object isn't JSON-serializable
                return json.dumps(str(obj), ensure_ascii=False)
            except Exception:
                return "[]"


# Optional: allow running this module directly for a quick test loop
if __name__ == "__main__":
    helper = LLMHelper()  # will auto-detect CSV and prompt for API key if needed
    print("Vacation Property Bot (type 'exit' to quit)")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Have a great vacation!")
            break

        if prompt.lower() == "exit":
            print("Bot: Have a great vacation!")
            break

        result = helper.search(prompt)
        if isinstance(result, dict) and "error" in result:
            print("Bot (error):", result.get("error"))
            details = result.get("details") or result.get("raw")
            if details:
                print("Details:", (details if isinstance(details, str) else str(details))[:240])
        else:
            tags = result.get("tags", []) if isinstance(result, dict) else []
            prop_ids = result.get("property_ids", []) if isinstance(result, dict) else []
            print("Bot: tags =", tags)
            print("Bot: property_ids =", prop_ids)
