# browser_manager.py
import logging
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


class BrowserManager:
    def __init__(self, headless=True, launch_args=None):
        self.headless = headless
        self.launch_args = launch_args or ["--no-sandbox", "--disable-dev-shm-usage"]
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.last_cookies = []
        self.last_url = None

    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless, args=self.launch_args)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
        except Exception as e:
            logger.debug("Error closing Playwright: %s", e)
        finally:
            if self.playwright:
                try:
                    self.playwright.stop()
                except Exception:
                    pass

    def fetch_page_content(self, url, timeout=60000):
        """
        Navigate to URL, wait for JS to render and return:
          { "html", "url", "cookies", "response_headers" }
        """
        try:
            logger.info("Playwright navigating to %s", url)
            response = self.page.goto(url, wait_until="networkidle", timeout=timeout)
            try:
                self.page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightTimeoutError:
                # ok if it times out, continue
                pass

            html = self.page.content()
            final_url = self.page.url
            cookies = self.context.cookies()
            self.last_cookies = cookies
            self.last_url = final_url
            headers = response.headers if response else {}

            return {
                "html": html,
                "url": final_url,
                "cookies": cookies,
                "response_headers": headers,
            }
        except Exception as e:
            logger.exception("Error fetching page content: %s", e)
            raise

    def _cookies_to_dict(self):
        d = {}
        for c in self.last_cookies or []:
            d[c.get("name")] = c.get("value")
        return d

    def _normalize_url(self, url):
        if not url:
            return url
        if urlparse(url).scheme:
            return url
        # relative -> join with last_url
        base = self.last_url or ""
        return urljoin(base, url)

    def download_file(self, url, save_path, timeout=30, retries=2):
        """
        Try to download a file robustly.
        Strategy:
          1) Normalize URL
          2) Try requests.get with Playwright cookies (streaming)
          3) If requests fails or returns HTML, fallback to Playwright page.goto() and save response.body()
          4) Retries for transient network errors
        Returns True on success, False otherwise.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        url = self._normalize_url(url)

        last_exc = None
        for attempt in range(1, retries + 2):
            try:
                # Strategy 1: requests with cookies
                logger.info("Attempting requests download (attempt %d) for %s", attempt, url)
                session = requests.Session()
                cookies = self._cookies_to_dict()
                if cookies:
                    session.cookies.update(cookies)
                # set small headers to appear like browser
                headers = {"User-Agent": "Mozilla/5.0"}
                with session.get(url, stream=True, timeout=timeout, headers=headers) as r:
                    r.raise_for_status()
                    ct = r.headers.get("Content-Type", "").lower()
                    # If it's HTML and not an attachment, treat that as failure
                    if "text/html" in ct and "attachment" not in (r.headers.get("Content-Disposition") or ""):
                        logger.info("Requests returned HTML (Content-Type), will fallback to Playwright")
                        raise ValueError("requests returned HTML - not raw file")
                    with open(save_path, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                fh.write(chunk)
                logger.info("Downloaded via requests to %s", save_path)
                return True
            except Exception as e:
                logger.debug("Requests attempt %d failed: %s", attempt, e)
                last_exc = e

            # Fallback to Playwright
            try:
                logger.info("Fallback to Playwright page.goto() for %s (attempt %d)", url, attempt)
                page = self.context.new_page()
                response = page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                if response and 200 <= (response.status or 0) < 400:
                    body = response.body()
                    if body:
                        with open(save_path, "wb") as fh:
                            fh.write(body)
                        logger.info("Playwright saved file to %s", save_path)
                        page.close()
                        return True
                    else:
                        logger.debug("Playwright response body empty for %s", url)
                else:
                    logger.debug("Playwright response status %s for %s", response.status if response else None, url)
                page.close()
            except Exception as e:
                logger.exception("Playwright fallback failed for %s: %s", url, e)
                last_exc = e

            # Wait a bit before retrying
            time.sleep(1)

        logger.error("All download attempts failed for %s: last_exc=%s", url, last_exc)
        return False
