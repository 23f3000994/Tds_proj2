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
    def __init__(self):
        self.ai_endpoint = getattr(Config, "AI_ENDPOINT", None)
        self.model = getattr(Config, "MODEL", None)

    def solve_quiz(self, page_content, downloaded_files):
        """
        page_content is a dict from BrowserManager.fetch_page_content:
          { "html": ..., "url": final_url, "cookies": ..., "response_headers": ... }
        downloaded_files is list of local file paths (may be empty).
        Returns dict with keys: submit_url, answer, reasoning (optional)
        """
        html = page_content.get("html", "") or ""
        base_url = page_content.get("url", "") or ""

        # find submit_url
        submit_url = self._find_submit_url(html, base_url)
        logger.info("LLMSolver found submit_url: %s", submit_url)

        # attempt to extract secret from page or files
        secret = None
        if "scrape" in base_url:
            secret = self._extract_secret_from_html(html) or self._extract_secret_from_files(downloaded_files)
        if secret:
            return {"submit_url": submit_url, "answer": secret, "reasoning": "secret extracted locally"}


        # Try to compute answers from downloaded files
        for path in downloaded_files or []:
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext in [".csv", ".tsv"]:
                    logger.info("Parsing CSV %s", path)
                    df = pd.read_csv(path)
                    ans = self._compute_best_numeric_answer(df)
                    if ans is not None:
                        return {"submit_url": submit_url, "answer": ans, "reasoning": f"computed from {os.path.basename(path)}"}
                elif ext in [".xls", ".xlsx"]:
                    logger.info("Parsing Excel %s", path)
                    df = pd.read_excel(path)
                    ans = self._compute_best_numeric_answer(df)
                    if ans is not None:
                        return {"submit_url": submit_url, "answer": ans, "reasoning": f"computed from {os.path.basename(path)}"}
                elif ext == ".json":
                    logger.info("Parsing JSON %s", path)
                    with open(path, "r", encoding="utf8") as fh:
                        obj = json.load(fh)
                    numeric = self._extract_numeric_from_json(obj)
                    if numeric is not None:
                        return {"submit_url": submit_url, "answer": numeric, "reasoning": f"computed from {os.path.basename(path)}"}
                # plain text fallback: search for numbers or "answer"
                else:
                    with open(path, "r", encoding="utf8", errors="ignore") as fh:
                        txt = fh.read()
                    m = re.search(r'answer\s*[:=]\s*([0-9\.\-]+)', txt, re.IGNORECASE)
                    if m:
                        val = float(m.group(1))
                        return {"submit_url": submit_url, "answer": int(val) if val.is_integer() else val, "reasoning": f"found answer in {os.path.basename(path)}"}
            except Exception as e:
                logger.exception("Error processing file %s: %s", path, e)

        # fallback to AI if configured
        if self.ai_endpoint:
            logger.info("No local solution; falling back to AI endpoint")
            prompt = self._build_prompt(html, base_url, downloaded_files)
            ai_resp = self._call_ai(prompt)
            # Expect dict with submit_url & answer
            if isinstance(ai_resp, dict) and "answer" in ai_resp:
                resp_submit = ai_resp.get("submit_url") or submit_url
                if resp_submit and not urlparse(resp_submit).scheme:
                    resp_submit = urljoin(base_url, resp_submit)
                ai_resp["submit_url"] = resp_submit
                return ai_resp
            # If not parseable, return error structure
            return {"submit_url": submit_url, "answer": None, "reasoning": "AI returned unexpected response"}
        else:
            logger.error("No AI configured and no local solution found")
            return {"submit_url": submit_url, "answer": None, "reasoning": "no-solution"}

    # ---------- helpers ----------
    def _find_submit_url(self, html, base_url):
        # look for JSON key "submit_url"
        m = re.search(r'"submit_url"\s*:\s*"([^"]+)"', html)
        if m:
            return urljoin(base_url, m.group(1))
        # <form action="...">
        m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, re.I)
        if m:
            return urljoin(base_url, m.group(1))
        # href containing submit
        m = re.search(r'href=["\']([^"\']*submit[^"\']*)["\']', html, re.I)
        if m:
            return urljoin(base_url, m.group(1))
        # fallback
        return urljoin(base_url, "/submit")

    def _extract_secret_from_html(self, html):
        m = re.search(r'"secret"\s*:\s*"([^"]+)"', html)
        if m:
            return m.group(1)
        m = re.search(r'Secret(?:\s*[:\-]\s*)([A-Za-z0-9_\-]{4,})', html, re.I)
        if m:
            return m.group(1)
        return None

    def _extract_secret_from_files(self, files):
        for p in files or []:
            try:
                with open(p, "r", encoding="utf8", errors="ignore") as fh:
                    txt = fh.read()
                m = re.search(r'"secret"\s*:\s*"([^"]+)"', txt)
                if m:
                    return m.group(1)
                m = re.search(r'Secret(?:\s*[:\-]\s*)([A-Za-z0-9_\-]{4,})', txt, re.I)
                if m:
                    return m.group(1)
            except Exception:
                continue
        return None

    def _compute_best_numeric_answer(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return None
        if "value" in df.columns:
            s = pd.to_numeric(df["value"], errors="coerce").fillna(0).sum()
            return int(s) if float(s).is_integer() else float(s)
        col_sums = {c: df[c].astype("float64").fillna(0).abs().sum() for c in numeric_cols}
        best = max(col_sums, key=col_sums.get)
        s = df[best].astype("float64").fillna(0).sum()
        return int(s) if float(s).is_integer() else float(s)

    def _extract_numeric_from_json(self, obj):
        if isinstance(obj, list):
            if all(isinstance(x, (int, float)) for x in obj):
                return sum(obj)
            if all(isinstance(x, dict) for x in obj):
                vals = [x.get("value") for x in obj if isinstance(x.get("value"), (int, float))]
                if vals:
                    return sum(vals)
        elif isinstance(obj, dict):
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "value" and isinstance(v, (int, float)):
                            yield v
                        else:
                            yield from walk(v)
                elif isinstance(o, list):
                    for i in o:
                        yield from walk(i)
            total = 0
            found = False
            for n in walk(obj):
                total += n
                found = True
            if found:
                return total
        return None

    def _build_prompt(self, html, base_url, downloaded_files):
        snippet = (html or "")[:4000]
        files = [os.path.basename(p) for p in (downloaded_files or [])]
        prompt = (
            "Return strict JSON {submit_url, answer, reasoning}.\n"
            f"Page URL: {base_url}\n"
            "Page HTML snippet:\n"
            f"{snippet}\n"
            f"Downloaded files: {files}\n"
            "If you can compute the answer from the files, do so. Otherwise return the submit_url and answer."
        )
        return prompt

    def _call_ai(self, prompt):
        payload = {"model": self.model or "openai/gpt-4o-mini", "input": prompt}
        try:
            r = requests.post(self.ai_endpoint, json=payload, timeout=30)
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                text = r.text
                m = re.search(r'(\{.*\})', text, re.DOTALL)
                if m:
                    return json.loads(m.group(1))
                return {"error": "AI returned non-json", "raw": text}
        except Exception as e:
            logger.exception("AI call failed: %s", e)
            return {"error": str(e)}
