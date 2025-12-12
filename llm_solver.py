# llm_solver.py
import logging
import re
import json
import os
from urllib.parse import urljoin, urlparse
import pandas as pd

import requests
from config import Config

logger = logging.getLogger(__name__)


class LLMSolver:
    """
    Solve quizzes using local computation when possible, otherwise call LLM endpoint.

    solve_quiz(page_content: dict, downloaded_files: list[str]) -> dict:
        should return a dict containing at least:
            - "submit_url": absolute URL where to POST the result
            - "answer": the answer (number/string/json/file)
            - optional "reasoning": explanation string
    """

    def __init__(self):
        # Config should provide AI_ENDPOINT and MODEL info if fallback to LLM is needed.
        self.ai_endpoint = getattr(Config, "AI_ENDPOINT", None)
        self.model = getattr(Config, "MODEL", None)

    def solve_quiz(self, page_content, downloaded_files):
        """
        Main entry. Try local solutions first.
        page_content: dict returned by BrowserManager.fetch_page_content
        downloaded_files: list of file paths saved locally
        """
        html = page_content.get("html", "")
        base_url = page_content.get("url", "")

        # 1) find submit_url in page HTML (absolute or relative)
        submit_url = self._find_submit_url(html, base_url)
        logger.info("LLMSolver found submit_url: %s", submit_url)

        # 2) try to extract secret from HTML (or from downloaded files)
        secret_from_page = self._extract_secret_from_html(html)
        if secret_from_page:
            logger.info("Found secret in page HTML")
        else:
            secret_from_page = self._extract_secret_from_files(downloaded_files)
            if secret_from_page:
                logger.info("Found secret in downloaded files")

        # 3) If CSV/XLSX present, compute sensible numeric answers (sum of 'value' column or best guess)
        # This is crucial for tasks like "sum of value column".
        for path in downloaded_files:
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext in [".csv", ".tsv"]:
                    logger.info("Parsing CSV file locally: %s", path)
                    df = pd.read_csv(path)
                    answer = self._compute_best_numeric_answer(df)
                    if answer is not None:
                        logger.info("Computed local numeric answer: %s", answer)
                        return {"submit_url": submit_url, "answer": answer, "reasoning": "Computed from CSV locally"}
                if ext in [".xlsx", ".xls"]:
                    logger.info("Parsing Excel file locally: %s", path)
                    df = pd.read_excel(path)
                    answer = self._compute_best_numeric_answer(df)
                    if answer is not None:
                        logger.info("Computed local numeric answer: %s", answer)
                        return {"submit_url": submit_url, "answer": answer, "reasoning": "Computed from Excel locally"}
                if ext in [".json"]:
                    logger.info("Parsing JSON file locally: %s", path)
                    with open(path, "r", encoding="utf8") as fh:
                        obj = json.load(fh)
                    # attempt to find numeric arrays or keys
                    numeric = self._extract_numeric_from_json(obj)
                    if numeric is not None:
                        logger.info("Computed numeric answer from JSON: %s", numeric)
                        return {"submit_url": submit_url, "answer": numeric, "reasoning": "Computed from JSON locally"}
            except Exception as e:
                logger.exception("Error parsing local file %s: %s", path, e)

        # 4) If secret was detected, return it directly (many demo quizzes require posting back secret)
        if secret_from_page:
            # If the question asks to post the secret, return it.
            return {"submit_url": submit_url, "answer": secret_from_page, "reasoning": "Extracted secret from page/files"}

        # 5) Otherwise, fallback to LLM endpoint (structured prompt) if configured
        if self.ai_endpoint:
            logger.info("Falling back to AI endpoint for solving")
            prompt = self._build_prompt(page_content, downloaded_files)
            resp = self._call_ai(prompt)
            # Expecting resp to be JSON-like with fields: submit_url, answer, reasoning (like your earlier AIPipe responses)
            if isinstance(resp, dict) and "submit_url" in resp and "answer" in resp:
                # normalize submit_url relative to base_url
                resp_submit_url = resp.get("submit_url")
                if not resp_submit_url:
                    resp_submit_url = submit_url
                if resp_submit_url and not urlparse(resp_submit_url).scheme:
                    resp["submit_url"] = urljoin(base_url, resp_submit_url)
                return resp
            else:
                logger.error("AI endpoint returned unexpected response: %s", resp)
                # safe fallback: return error
                return {"submit_url": submit_url, "answer": None, "reasoning": "AI returned unexpected response"}
        else:
            logger.error("No AI endpoint configured and no local solution found")
            return {"submit_url": submit_url, "answer": None, "reasoning": "No solution available"}

    # ---------------- helper methods ----------------
    def _find_submit_url(self, html, base_url):
        """
        Look for a submit URL in the HTML. Handles absolute and relative URLs.
        Patterns:
          - JSON payload containing "submit"
          - <form action="...">
          - any href to /submit or submit
        """
        # 1) Try to find common JSON keys if page contains JSON in script
        m = re.search(r'"submit_url"\s*:\s*"([^"]+)"', html)
        if m:
            return urljoin(base_url, m.group(1))

        # 2) form action
        m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return urljoin(base_url, m.group(1))

        # 3) href that contains 'submit'
        m = re.search(r'href=["\']([^"\']*submit[^"\']*)["\']', html, re.IGNORECASE)
        if m:
            return urljoin(base_url, m.group(1))

        # 4) fallback: common endpoint
        return urljoin(base_url, "/submit")

    def _extract_secret_from_html(self, html):
        # Look for JSON-style secret or patterns like "secret": "..."
        m = re.search(r'"secret"\s*:\s*"([^"]+)"', html)
        if m:
            return m.group(1)
        # some pages put secret in clear text
        m = re.search(r'Secret(?:\s*[:\-]\s*)([A-Za-z0-9_\-]{4,})', html, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _extract_secret_from_files(self, files):
        for path in files:
            try:
                with open(path, "r", encoding="utf8", errors="ignore") as fh:
                    txt = fh.read()
                m = re.search(r'"secret"\s*:\s*"([^"]+)"', txt)
                if m:
                    return m.group(1)
                m = re.search(r'Secret(?:\s*[:\-]\s*)([A-Za-z0-9_\-]{4,})', txt, re.IGNORECASE)
                if m:
                    return m.group(1)
            except Exception:
                continue
        return None

    def _compute_best_numeric_answer(self, df: pd.DataFrame):
        """
        Heuristic numeric answer:
         - If column named 'value' use df['value'].sum()
         - Else find numeric columns and return the sum of the column with largest absolute sum
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return None

        if "value" in df.columns:
            s = pd.to_numeric(df["value"], errors="coerce").fillna(0).sum()
            return int(s) if float(s).is_integer() else float(s)

        # compute column sums and pick the most plausible
        col_sums = {c: df[c].astype("float64").fillna(0).abs().sum() for c in numeric_cols}
        # choose column with largest sum (heuristic)
        best_col = max(col_sums, key=col_sums.get)
        s = df[best_col].astype("float64").fillna(0).sum()
        return int(s) if float(s).is_integer() else float(s)

    def _extract_numeric_from_json(self, obj):
        """
        Try to find arrays or objects with numeric lists and return a sum or a sensible numeric answer.
        """
        if isinstance(obj, list):
            # if list of numbers
            if all(isinstance(x, (int, float)) for x in obj):
                return sum(obj)
            # if list of dicts, try to find numeric field 'value'
            if all(isinstance(x, dict) for x in obj):
                values = []
                for x in obj:
                    if "value" in x and isinstance(x["value"], (int, float)):
                        values.append(x["value"])
                if values:
                    return sum(values)
        elif isinstance(obj, dict):
            # search recursively for numeric arrays or 'value' keys
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "value" and isinstance(v, (int, float)):
                            yield v
                        else:
                            yield from walk(v)
                elif isinstance(o, list):
                    for item in o:
                        yield from walk(item)
            total = 0
            found = False
            for n in walk(obj):
                total += n
                found = True
            if found:
                return total
        return None

    def _build_prompt(self, page_content, downloaded_files):
        """
        Build a structured prompt for the AI to solve when local computation fails.
        Keep it explicit:
         - include page URL
         - include small snippet of page HTML (not entire huge pages)
         - list any downloaded files' filenames
         - ask for JSON output with keys: submit_url, answer, reasoning
        """
        html = page_content.get("html", "")
        url = page_content.get("url", "")

        snippet = (html or "")[:4000]  # cut to reasonable length
        file_list = [os.path.basename(p) for p in (downloaded_files or [])]

        prompt = (
            "You are given a quiz page and optional files. Respond with strict JSON {submit_url, answer, reasoning}.\n\n"
            f"Page URL: {url}\n\n"
            "Page HTML (snippet):\n"
            f"{snippet}\n\n"
            f"Downloaded files: {file_list}\n\n"
            "Task: extract the answer required by the page and provide the submit_url (absolute or relative). "
            "If files are present and contain data, compute numeric answers locally (preferred). "
            "Return only valid JSON."
        )
        return prompt

    def _call_ai(self, prompt):
        """
        Call configured AI endpoint. Expect JSON response.
        This function assumes Config.AI_ENDPOINT accepts JSON: {model:..., input:...}
        Adjust to your AIPipe contract.
        """
        payload = {
            "model": self.model or "openai/gpt-4o-mini",
            "input": prompt
        }
        try:
            r = requests.post(self.ai_endpoint, json=payload, timeout=30)
            r.raise_for_status()
            # Accept direct JSON or text that contains JSON
            try:
                return r.json()
            except ValueError:
                # attempt to extract JSON object from text
                text = r.text
                m = re.search(r'(\{.*\})', text, re.DOTALL)
                if m:
                    return json.loads(m.group(1))
                return {"error": "AI returned non-JSON response", "raw": text}
        except Exception as e:
            logger.exception("AI call failed: %s", e)
            return {"error": str(e)}
