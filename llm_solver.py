# llm_solver.py
"""
LLMSolver for the LLM Analysis Quiz project.

Responsibilities:
- Extract the submit_url from the quiz page HTML.
- Try local heuristics to solve common quiz types:
  * secret extraction (for "scrape" pages / pages containing a secret string)
  * CSV numeric sum (for downloaded CSV files, prefer column named "value")
- If local heuristics fail and an API key is present (OPENAI_API_KEY or AIPIPE_API_KEY),
  call the LLM to produce a JSON-like solution with fields:
    { "submit_url": "...", "answer": <value>, "reasoning": "..." }
- Return a dict containing at least "submit_url" and "answer" or raise ValueError.
"""

import os
import re
import json
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Any, Dict

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMSolver:
    def __init__(self, model: Optional[str] = None):
        """
        Prefers environment variables:
        - AIPIPE_API_KEY (preferred if you used AIPipe earlier)
        - OPENAI_API_KEY (fallback)
        Model can be provided; otherwise we default sensibly.
        """
        self.aipipe_key = os.getenv("AIPIPE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        logger.info("LLMSolver configured. AIPipe=%s OpenAI=%s model=%s",
                    bool(self.aipipe_key), bool(self.openai_key), self.model)

    # ----------------------
    # Public API
    # ----------------------
    def solve_quiz(self, page_content: Dict[str, Any], downloaded_files: List[str]) -> Dict[str, Any]:
        """
        Attempt to solve quiz using local heuristics first; fallback to LLM if configured.

        page_content expected keys:
          - 'html' : HTML string (prefer)
          - 'url'  : original page URL (helpful for relative submit URLs)
          - (optional) 'text' : plain text extracted by browser_manager (may be absent)

        downloaded_files: list of local file paths (CSV/PDF/etc) that browser_manager has saved.
        """
        html = page_content.get("html", "") or ""
        page_url = page_content.get("url") or page_content.get("page_url") or ""
        text = page_content.get("text")  # may be missing; don't crash

        logger.info("LLMSolver.solve_quiz: page_url=%s files=%d", page_url, len(downloaded_files))

        # 1) find submit_url in HTML
        submit_url = self._find_submit_url(html, page_url)
        if not submit_url:
            # fallback: maybe submit_url encoded in text
            submit_url = self._find_submit_url_from_text(text or "", page_url)

        if not submit_url:
            logger.warning("No submit_url found in page content; will still try heuristics and LLM.")
        else:
            logger.info("LLMSolver found submit_url: %s", submit_url)

        # 2) Try local heuristics (fast, deterministic)
        # 2.a Secret extraction (for "scrape" style pages). Only do this when page_url or html indicates a secret-type quiz.
        secret_answer = None
        if self._looks_like_secret_quiz(page_url, html):
            secret_answer = self._extract_secret_from_html_or_files(html, downloaded_files)
            if secret_answer is not None:
                logger.info("LLMSolver extracted secret locally: %s", repr(secret_answer))
                return {"submit_url": submit_url, "answer": secret_answer, "reasoning": "Secret extracted locally"}

        # 2.b CSV numeric sum (if a CSV file was downloaded)
        csv_answer = self._sum_csv_from_files(downloaded_files)
        if csv_answer is not None:
            logger.info("LLMSolver computed CSV-sum locally: %s", csv_answer)
            return {"submit_url": submit_url, "answer": csv_answer, "reasoning": "CSV sum computed locally"}

        # 3) If local heuristics didn't help, call LLM (if configured)
        if self.aipipe_key or self.openai_key:
            try:
                logger.info("No local solution — will call remote LLM (model=%s).", self.model)
                llm_response = self._call_llm_solver(html=html, text=text, page_url=page_url, files=downloaded_files)
                # Expect LLM to return a JSON-like dict with 'submit_url' and 'answer' fields.
                if isinstance(llm_response, dict):
                    # normalize submit_url (LLM may return relative)
                    if 'submit_url' in llm_response and page_url:
                        llm_response['submit_url'] = self._normalize_submit_url(llm_response['submit_url'], page_url)
                    return llm_response
                else:
                    logger.warning("LLM returned non-dict response; wrapping into answer field.")
                    return {"submit_url": submit_url, "answer": llm_response, "reasoning": "LLM raw output"}
            except Exception as e:
                logger.exception("LLM call failed: %s", e)
                raise

        # 4) No solution possible
        raise ValueError("Unable to solve quiz locally and no LLM configured.")

    # ----------------------
    # Helper: find submit url
    # ----------------------
    def _find_submit_url(self, html: str, page_url: str) -> Optional[str]:
        """
        Look for common patterns of submit endpoints in the HTML.
        - data-submit-url="..."
        - fetch('/submit'...) or "/submit" inside scripts
        - JSON blocks containing "submit_url"
        Returns normalized absolute URL if possible.
        """
        if not html:
            return None

        # 1) JSON field: "submit_url": "..."
        m = re.search(r'["\']submit_url["\']\s*:\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # 2) data attribute
        m = re.search(r'data-submit-url=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # 3) simple /submit usage inside JS
        m = re.search(r'["\'](\/submit(?:[^\s"\'>]*)?)["\']', html)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # 4) form action
        m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        return None

    def _find_submit_url_from_text(self, text: str, page_url: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r'(https?://[^\s"\'<>]+/submit[^\s"\'<>]*)', text)
        if m:
            return m.group(1)
        m = re.search(r'(/submit[^\s"\'<>]*)', text)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)
        return None

    def _normalize_submit_url(self, candidate: Optional[str], page_url: str) -> Optional[str]:
        if not candidate:
            return None
        candidate = candidate.strip()
        # If it's already absolute:
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
        # if it's scheme-less (e.g. //example.com/submit)
        if candidate.startswith("//"):
            parsed = urlparse(page_url or "https://example.com")
            return f"{parsed.scheme}:{candidate}"
        # otherwise join with page_url
        try:
            base = page_url or ""
            absolute = urljoin(base, candidate)
            return absolute
        except Exception:
            return candidate

    # ----------------------
    # Heuristic: detect secret quizzes & extract secret
    # ----------------------
    def _looks_like_secret_quiz(self, page_url: str, html: str) -> bool:
        """
        Decide whether the quiz is of the "secret/scrape" type.
        We conservatively check:
         - page_url contains '/scrape' OR
         - html contains "secret" near some JSON-looking structure
        """
        if page_url and "scrape" in page_url:
            return True
        if html and re.search(r'["\']secret["\']\s*:', html, re.IGNORECASE):
            return True
        return False

    def _extract_secret_from_html_or_files(self, html: str, files: List[str]) -> Optional[str]:
        # try HTML first: look for JSON or assignments like "secret": "your secret"
        if html:
            m = re.search(r'["\']secret["\']\s*:\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
            if m:
                return m.group(1)

            # sometimes secret is in base64 inside a script (common in sample quizzes)
            # find base64 block in innerHTML assignments and try to decode patterns that contain "secret"
            b64_matches = re.findall(r'atob\(\s*["\']([A-Za-z0-9+/=\s]+)["\']\s*\)', html)
            for b64 in b64_matches:
                try:
                    import base64
                    decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                    m2 = re.search(r'["\']secret["\']\s*:\s*["\']([^"\']+)["\']', decoded, re.IGNORECASE)
                    if m2:
                        return m2.group(1)
                except Exception:
                    continue

        # fallback: search downloaded files for a secret-like field (simple JSON or text)
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    m = re.search(r'["\']secret["\']\s*:\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                    if m:
                        return m.group(1)
                    # or plain "secret: abc"
                    m2 = re.search(r'\bsecret\s*[:=]\s*([A-Za-z0-9_\-@]+)', content, re.IGNORECASE)
                    if m2:
                        return m2.group(1)
            except Exception:
                continue
        return None

    # ----------------------
    # Heuristic: sum CSV(s)
    # ----------------------
    def _sum_csv_from_files(self, files: List[str]) -> Optional[Any]:
        """
        If any downloaded file is CSV, try to compute a sensible numeric answer.
        Preference:
         - If a column named 'value' or 'Value' or 'amount' exists, sum it.
         - Otherwise, sum all numeric cells.
        Returns numeric (int if integral) or None.
        """
        csv_paths = [p for p in files if p.lower().endswith(".csv")]
        if not csv_paths:
            return None

        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                logger.warning("Failed to read CSV %s: %s", path, e)
                continue

            if df.empty:
                continue

            # Prefer column names
            for candidate in ["value", "Value", "amount", "Amount", "values"]:
                if candidate in df.columns:
                    try:
                        s = pd.to_numeric(df[candidate], errors="coerce").sum(skipna=True)
                        return self._coerce_number(s)
                    except Exception:
                        pass

            # If numeric columns exist, sum them
            numeric = df.select_dtypes(include=["number"])
            if not numeric.empty:
                try:
                    s = numeric.sum().sum()  # sum across dataframe
                    return self._coerce_number(s)
                except Exception:
                    pass

            # last resort: try to parse any numeric tokens in the CSV as fallback
            try:
                all_text = df.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
                nums = re.findall(r'-?\d+(?:\.\d+)?', all_text)
                nums = [float(n) for n in nums]
                if nums:
                    return self._coerce_number(sum(nums))
            except Exception:
                pass

        return None

    def _coerce_number(self, val: Any) -> Any:
        """Return int if val is effectively integer, else float"""
        try:
            if val is None:
                return None
            if float(val).is_integer():
                return int(round(float(val)))
            return float(val)
        except Exception:
            return val

    # ----------------------
    # LLM call (fallback)
    # ----------------------
    def _call_llm_solver(self, html: str, text: Optional[str], page_url: str, files: List[str]) -> Dict[str, Any]:
        """
        Build a concise prompt with the HTML (trimmed) and info about files, then call remote LLM.
        This function prefers AIPipe (AIPIPE_API_KEY) if present; otherwise tries OPENAI_API_KEY.
        The prompt asks for a JSON with keys: submit_url, answer, reasoning.
        """
        # Build prompt
        short_html = (html or "")[:4000]  # keep prompt bounded
        prompt = (
            "You are given a web quiz page (HTML snippet) and a list of downloaded files.\n"
            "Please extract the URL where the answer must be POSTed (submit_url) and produce the answer.\n"
            "Return ONLY a JSON object with keys: submit_url (string), answer (string/number/object), reasoning (string).\n\n"
            "Page URL: " + (page_url or "") + "\n\n"
            "HTML_SNIPPET:\n" + short_html + "\n\n"
            "TEXT_SNIPPET:\n" + (text or "")[:1000] + "\n\n"
            "FILES:\n" + ", ".join([os.path.basename(f) for f in files]) + "\n\n"
            "If a CSV was downloaded, compute sums exactly. If the submit_url is relative, return it as given (we will normalize)."
        )

        # Call proper API
        if self.aipipe_key:
            return self._call_openai_chat_api(prompt, api_key=self.aipipe_key, use_openai_endpoint=False)
        if self.openai_key:
            return self._call_openai_chat_api(prompt, api_key=self.openai_key, use_openai_endpoint=True)

        raise RuntimeError("No LLM API key available")

    def _call_openai_chat_api(self, prompt: str, api_key: str, use_openai_endpoint: bool = True) -> Dict[str, Any]:
        """
        Minimal call to an LLM endpoint. If use_openai_endpoint is True, call OpenAI Chat Completions.
        If False, we attempt a generic "AIPipe" compatible call using the same OpenAI-style endpoint.
        This function expects JSON result in text — we'll try to parse JSON from the response.
        """
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # We will attempt OpenAI v1 chat/completions for compatibility
        if use_openai_endpoint:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that outputs exactly a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.0
            }
        else:
            # Try same shape for other providers; many accept similar shape. If your AIPipe endpoint differs,
            # replace this with the correct client call.
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that outputs exactly a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.0
            }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logger.error("LLM API returned %s: %s", resp.status_code, resp.text)
            raise RuntimeError(f"LLM API error: {resp.status_code} {resp.text}")

        data = resp.json()
        # Try to extract textual reply from common shapes
        text_out = None
        if "choices" in data and len(data["choices"]) > 0:
            # Chat completions shape
            text_out = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text")
        elif "output" in data:
            # other shapes
            if isinstance(data["output"], list):
                text_out = " ".join([str(x) for x in data["output"]])
            else:
                text_out = str(data["output"])
        else:
            # fallback: whole JSON
            text_out = json.dumps(data)

        # Extract JSON object from text_out
        try:
            # find first {...} block
            jmatch = re.search(r'(\{[\s\S]*\})', text_out)
            if jmatch:
                jtxt = jmatch.group(1)
                result = json.loads(jtxt)
                # cast answer numeric strings to numbers if possible
                if 'answer' in result:
                    result['answer'] = self._try_cast_numeric(result['answer'])
                return result
            # else if the model returned raw value (like "12345")
            # try to parse as a number
            stripped = text_out.strip()
            if re.fullmatch(r'-?\d+', stripped):
                return {"submit_url": None, "answer": int(stripped), "reasoning": "LLM returned a raw integer"}
            if re.fullmatch(r'-?\d+\.\d+', stripped):
                return {"submit_url": None, "answer": float(stripped), "reasoning": "LLM returned a raw float"}
            # fallback return raw text
            return {"submit_url": None, "answer": stripped, "reasoning": "LLM returned unstructured text"}
        except Exception as e:
            logger.exception("Failed to parse LLM text output: %s", e)
            raise

    def _try_cast_numeric(self, val):
        # if val is string representing an integer/float, cast it
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            s = val.strip().replace(",", "")
            if re.fullmatch(r'-?\d+', s):
                return int(s)
            if re.fullmatch(r'-?\d+\.\d+', s):
                return float(s)
        return val
