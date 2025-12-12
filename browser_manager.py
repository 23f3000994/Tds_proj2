# browser_manager.py
import logging
import os
from urllib.parse import urlparse
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


class BrowserManager:
    """
    Context manager for Playwright browser.
    Methods:
      - fetch_page_content(url) -> dict with keys: html, url, cookies, response_headers
      - download_file(url, save_path, timeout=60) -> True/False
    """

    def __init__(self, headless=True, launch_args=None):
        self.headless = headless
        self.launch_args = launch_args or ["--no-sandbox", "--disable-dev-shm-usage"]
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.last_cookies = []  # store cookies for requests fallback

    def __enter__(self):
        self.playwright = sync_playwright().start()
        # Use Chromium (Playwright official image has browsers installed)
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
            logger.debug("Error closing Playwright objects: %s", e)
        finally:
            if self.playwright:
                try:
                    self.playwright.stop()
                except Exception:
                    pass

    def fetch_page_content(self, url, timeout=60000):
        """
        Visit URL with Playwright and return page HTML and metadata.
        Returns:
            {
              "html": "<html>...</html>",
              "url": final_url,
              "cookies": [...],
              "response_headers": {...}
            }
        """
        try:
            logger.info("Playwright navigating to %s", url)
            response = self.page.goto(url, wait_until="networkidle", timeout=timeout)
            # ensure we wait small amount to allow JS to render dynamic content
            try:
                self.page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightTimeoutError:
                # sometimes networkidle never occurs, continue
                pass

            html = self.page.content()
            final_url = self.page.url
            cookies = self.context.cookies()
            self.last_cookies = cookies
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

    def _cookies_to_requests_dict(self):
        jar = {}
        for c in self.last_cookies:
            jar[c.get("name")] = c.get("value")
        return jar

    def download_file(self, url, save_path, timeout=60):
        """
        Try to download a file. Strategy:
         1) Try plain requests.get with cookies extracted from last page (fast & reliable).
         2) If that fails or returns HTML, fallback to Playwright: open a new page and save response body.
        Returns True if saved, False otherwise.
        """
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Strategy 1: requests with cookies (fast)
        try:
            logger.info("Attempting HTTP download via requests: %s", url)
            session = requests.Session()
            # set cookies from context if available
            cookies = self._cookies_to_requests_dict()
            if cookies:
                session.cookies.update(cookies)

            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # optionally check content-type; if it's HTML, treat as failure
                ct = r.headers.get("Content-Type", "")
                if "text/html" in ct.lower() and r.headers.get("Content-Disposition") is None:
                    logger.info("Requests returned HTML content-type; falling back to Playwright download")
                    raise ValueError("requests returned HTML - likely page, not raw file")

                # write to file
                with open(save_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            logger.info("Downloaded file via requests to %s", save_path)
            return True
        except Exception as e:
            logger.debug("Requests download failed (%s), falling back to Playwright: %s", url, e)

        # Strategy 2: Use Playwright's page to navigate and save response body
        try:
            logger.info("Attempting Playwright download fallback for: %s", url)
            # open a fresh temporary page so we don't clobber main page state
            page = self.context.new_page()
            # navigate and capture response
            response = page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            if response is None:
                logger.error("Playwright goto returned None for %s", url)
                page.close()
                return False

            # if response is OK and has binary body, write it
            status = response.status
            if status and 200 <= status < 400:
                body = response.body()
                if body:
                    with open(save_path, "wb") as fh:
                        fh.write(body)
                    logger.info("Playwright saved file to %s", save_path)
                    page.close()
                    return True
                else:
                    logger.error("Playwright response body empty for %s", url)
            else:
                logger.error("Playwright response status %s for %s", status, url)
            page.close()
            return False
        except Exception as e:
            logger.exception("Playwright download fallback failed for %s: %s", url, e)
            return False
