# LLMhelper.py
"""
LLMHelper: OpenRouter LLM integration that reads properties from a **CSV**,
injects them into the prompt as JSON, and returns structured tags / property IDs.

Usage in your project:
    from LLMhelper import LLMHelper
    llm = LLMHelper(csv_path="data/properties.csv")  # or leave None to auto-detect
    result = llm.search('Looking for a beach house for 4 under $250/night')
    print(result)  # {"tags": [...], "property_ids": [...]}

This class is drop-in safe and does **not** modify other parts of your code.
It includes a compatibility shim (see recommender/llm.py) so that
`from recommender.llm import LLMHelper` continues to work.
"""

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
        self.timeout = request_timeout
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or self._prompt_api_key()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.properties = self._load_properties_csv(csv_path)

    # -------------------------------
    # Public API
    # -------------------------------
    def search(self, user_prompt: str) -> Dict[str, Any]:
        """
        Call the LLM and return a parsed dict.
        On error, returns {'error': '...', 'details'?: ... , 'raw'?: ...}
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {
                    "role": "user",
                    "content": (
                            "PROPERTIES:\n" + json.dumps(self.properties, ensure_ascii=False) +
                            "\n\nUSER REQUEST:\n" + user_prompt +
                            "\n\nRespond with JSON: {\"tags\": [...], \"property_ids\": [...]} (property_ids optional)"
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
        try:
            r = requests.post(OPENROUTER_URL, headers=self.headers, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}", "details": r.text}
            data = r.json()
            msg = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not msg:
                return {"error": "No content in response", "details": data}
            if not expect_json:
                return msg

            # Try strict JSON parse
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
                        return {"error": "Model returned non-JSON content", "raw": msg}
                return {"error": "Model returned non-JSON content", "raw": msg}
        except Exception as e:
            return {"error": "Request failed", "details": str(e)}

    def _prompt_api_key(self) -> str:
        try:
            return getpass.getpass("Enter your OpenRouter API key: ").strip()
        except Exception:
            # Fallback (e.g., on some IDEs that cannot hide input)
            return input("Enter your OpenRouter API key: ").strip()

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
            if os.path.isfile(p):
                csv_file = p
                break

        if csv_file is None:
            raise FileNotFoundError(
                "Could not find a properties CSV. "
                "Tried: " + ", ".join(candidate_paths)
            )

        df = pd.read_csv(csv_file)

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Ensure there's a stable integer `property_id`
        if "property_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "property_id"})
            df["property_id"] = df["property_id"].astype(int)

        # Basic type cleanup
        if "nightly_price" in df.columns:
            df["nightly_price"] = pd.to_numeric(df["nightly_price"], errors="coerce")

        # Convert NaN to empty strings for JSON friendliness
        records = df.fillna("").to_dict(orient="records")
        return records


# Optional: allow running this module directly for a quick test loop
if __name__ == "__main__":
    helper = LLMHelper()  # will auto-detect CSV and prompt for API key if needed
    print("Vacation Property Bot (type 'exit' to quit)")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            print("Bot: Have a great vacation!")
            break
        result = helper.search(prompt)
        if isinstance(result, dict) and "error" in result:
            print("Bot (error):", result["error"])
            if "details" in result:
                print("Details:", result["details"][:240])
            elif "raw" in result:
                print("Raw output:", result["raw"][:240])
        else:
            tags = result.get("tags", []) if isinstance(result, dict) else []
            prop_ids = result.get("property_ids", []) if isinstance(result, dict) else []
            print("Bot: tags =", tags)
            print("Bot: property_ids =", prop_ids)
