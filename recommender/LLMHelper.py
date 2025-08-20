# recommender/LLMHelper.py
# OpenRouter helper with robust 429 handling, multiple model fallbacks,
# and a structured "extract_hints" endpoint that returns fields aligned
# to your dataset: tags, features, locations, environments, property_ids.

import os, time, json, requests
from typing import Any, Dict, List, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class LLMHelper:
    def __init__(
            self,
            api_key: Optional[str] = None,
            csv_path: Optional[str] = None,    # optional; you don't need to pass the whole CSV
            request_timeout: int = 20,
            model: Optional[str] = None,
            app_url: str = "http://localhost",
            app_title: str = "StayFinder",
    ):
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        self.csv_path = csv_path
        self.request_timeout = int(request_timeout)

        # Try in order; first available wins
        self.model_candidates = [
            model or "deepseek/deepseek-chat",                 # paid/standard
            "deepseek/deepseek-r1:free",                      # free pool alt
            "deepseek/deepseek-chat-v3-0324:free",            # legacy free pool
        ]

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # These improve your RL bucket at OpenRouter; keep them set
            "HTTP-Referer": app_url,
            "X-Title": app_title,
        }

    # -------- Internal HTTP helpers --------
    def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(OPENROUTER_URL, headers=self.headers, json=payload, timeout=self.request_timeout)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}", "details": r.text, "headers": dict(r.headers)}

    def _post_with_retry_and_fallback(self, base_payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        backoff = 0.7
        last_err = None
        for model in self.model_candidates:
            payload = dict(base_payload)
            payload["model"] = model
            for _ in range(max_retries):
                data = self._post_once(payload)
                if "error" not in data:
                    return data
                last_err = data
                err = data["error"]
                if err.startswith("HTTP 429"):
                    ra = data.get("headers", {}).get("Retry-After")
                    wait = float(ra) if ra else backoff
                    time.sleep(wait)
                    backoff = min(backoff * 2, 8.0)
                    continue
                break  # non-429 → try next model
        return last_err or {"error": "Unknown error"}

    # -------- Public high-level methods --------
    def extract_hints(self, user_free_text: str) -> Dict[str, Any]:
        """
        Parse the user's free-text and return dataset-aligned hints:
        {
          "tags": [str], "features": [str],
          "locations": [str], "environments": [str],
          "property_ids": [str]   # optional
        }
        Always JSON. Keep it compact (few items per list).
        """
        system = (
            "You extract search hints for vacation rentals. "
            "Return strict JSON keys: tags, features, locations, environments, property_ids. "
            "Each value must be a list of short strings that are likely to match column values "
            "in a dataset (columns like location, environment, property_type, features, tags). "
            "Do not include explanations or extra keys. Keep lists short (<=6)."
        )
        payload = {
            "temperature": 0.2,
            "max_tokens": 160,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"User text:\n{user_free_text}\n\nRespond only with valid JSON."},
            ],
        }
        data = self._post_with_retry_and_fallback(payload)
        if "error" in data:
            return {"error": data["error"], "details": data.get("details", "")}
        msg = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        try:
            obj = json.loads(msg)
        except Exception:
            return {"error": "Model returned non-JSON content", "raw": msg}

        # Normalize output
        def _listify(x):
            if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
            if isinstance(x, str) and x.strip(): return [x.strip()]
            return []

        return {
            "tags": _listify(obj.get("tags")),
            "features": _listify(obj.get("features")),
            "locations": _listify(obj.get("locations")),
            "environments": _listify(obj.get("environments")),
            "property_ids": [str(i).strip() for i in (obj.get("property_ids") or []) if str(i).strip()],
        }

    def generate_travel_blurb(self, prompt: str) -> str:
        payload = {
            "temperature": 0.3,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "Write a short, upbeat 2–3 sentence travel blurb. No markdown."},
                {"role": "user", "content": prompt},
            ],
        }
        data = self._post_with_retry_and_fallback(payload)
        if "error" in data:
            return f"ERROR: {data['error']}: {data.get('details','')}"
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or "ERROR: empty"
