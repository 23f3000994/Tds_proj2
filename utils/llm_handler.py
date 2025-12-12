import re
import json
import logging
from urllib.parse import urljoin

logger = logging.getLogger("llm_solver")


class LLMSolver:
    def __init__(self, ai_client=None, model_name="gpt-4o-mini"):
        """
        ai_client: optional callable for LLM usage (e.g., OpenAI, AIPipe)
        model_name: string name of model to use
        """
        self.ai_client = ai_client
        self.model_name = model_name

    # ------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------
    def solve_quiz(self, page_content: dict, files: list):
        """
        page_content contains:
            - url
            - html
            - text
        files contains list of (filename, path)
        """

        page_url = page_content.get("url", "")
        html = page_content.get("html", "")
        text = page_content.get("text", "") or ""

        logger.info(f"LLMSolver.solve_quiz: page_url={page_url} files={len(files)}")

        # Try to find submit_url from page HTML
        submit_url = self._find_submit_url(html, page_url)
        if submit_url:
            logger.info(f"LLMSolver found submit_url: {submit_url}")
        else:
            logger.warning("No submit_url found in page content; will still try heuristics and LLM.")

        # SECRET QUIZ DETECTION - ENABLE ONLY FOR SCRAPE PAGES
        if "scrape" in page_url:
            secret = self._extract_secret(html)
            if secret:
                logger.info(f"Extracted secret: {secret}")
                return {
                    "submit_url": submit_url,
                    "answer": secret,
                    "reasoning": "Secret quiz detected; extracted from page"
                }

        # AUDIO QUIZ HANDLING
        if files:
            logger.info("Audio/Data quiz detected, processing CSV...")
            answer = self._solve_csv_sum(files)
            return {
                "submit_url": submit_url,
                "answer": answer,
                "reasoning": "Calculated sum from CSV"
            }

        # FALLBACK: USE AI CLIENT
        if self.ai_client:
            logger.info("Using LLM client...")
            return self._ask_ai(html, submit_url)
        else:
            logger.error("No AI configured and no local solution found")
            return {"submit_url": submit_url, "answer": None}

    # ------------------------------------------------------
    # SUBMIT URL EXTRACTION
    # ------------------------------------------------------
    def _find_submit_url(self, html: str, page_url: str):
        if not html:
            return None

        # JSON pattern: "submit_url": "/submit"
        m = re.search(r'["\']submit_url["\']\s*:\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # JavaScript var: const SUBMIT_URL = "/submit"
        m = re.search(r'SUBMIT_URL\s*=\s*["\']([^"\']+)["\']', html)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # data attribute
        m = re.search(r'data-submit-url=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # Direct "/submit"
        m = re.search(r'["\'](\/submit[^"\']*)["\']', html)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        # Form action
        m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return self._normalize_submit_url(m.group(1), page_url)

        return None

    # ------------------------------------------------------
    def _normalize_submit_url(self, raw_url, page_url):
        if raw_url.startswith("http"):
            return raw_url
        return urljoin(page_url, raw_url)

    # ------------------------------------------------------
    # SECRET EXTRACTION
    # ------------------------------------------------------
    def _extract_secret(self, html: str):
        if not html:
            return None

        # look for JS variable or JSON field containing "secret"
        m = re.search(r'["\']secret["\']\s*:\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.search(r'SECRET\s*=\s*["\']([^"\']+)["\']', html)
        if m:
            return m.group(1)

        return None

    # ------------------------------------------------------
    # CSV / AUDIO QUIZ SOLVER
    # ------------------------------------------------------
    def _solve_csv_sum(self, files):
        import csv

        # Expect exactly one CSV file
        fname, fpath = files[0]

        with open(fpath, "r") as f:
            reader = csv.reader(f)
            nums = []
            for row in reader:
                for x in row:
                    try:
                        nums.append(float(x))
                    except:
                        pass

        return int(sum(nums))

    # ------------------------------------------------------
    # LLM FALLBACK
    # ------------------------------------------------------
    def _ask_ai(self, html, submit_url):
        logger.info("Asking LLM as fallback...")

        prompt = f"""
You are solving a quiz. The HTML content is below:

{html}

Your job:
1. Identify the correct submit_url (if missing, output /submit)
2. Determine the correct answer.

Return a JSON dictionary with:
- "submit_url"
- "answer"
"""

        resp = self.ai_client(prompt, model=self.model_name)

        try:
            return json.loads(resp)
        except:
            return {"submit_url": submit_url, "answer": None}
