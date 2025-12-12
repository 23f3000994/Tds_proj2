import requests
import logging
import os
import re
import time
from urllib.parse import urljoin, urlparse

from browser_manager import BrowserManager
from llm_solver import LLMSolver
from config import Config

logger = logging.getLogger(__name__)


class QuizProcessor:
    def __init__(self, email, secret):
        self.email = email
        self.secret = secret
        self.llm_solver = LLMSolver()
        self.temp_dir = 'temp_downloads'
        os.makedirs(self.temp_dir, exist_ok=True)

    def process_quiz_chain(self, initial_url):
        """Process a chain of quizzes starting from initial_url"""
        current_url = initial_url
        start_time = time.time()
        attempt_count = 0
        max_attempts = 10  # Prevent infinite loops

        while current_url and attempt_count < max_attempts:
            attempt_count += 1
            elapsed = time.time() - start_time

            if elapsed > Config.QUIZ_TIMEOUT:
                logger.error("Quiz timeout exceeded")
                return {"error": "Timeout exceeded"}

            logger.info(f"Processing quiz {attempt_count}: {current_url}")

            try:
                result = self.solve_single_quiz(current_url)

                if result.get('correct'):
                    logger.info(f"Quiz {attempt_count} solved correctly!")
                    next_url = result.get('url')
                    if next_url:
                        logger.info(f"Moving to next quiz: {next_url}")
                        current_url = next_url
                    else:
                        logger.info("Quiz chain completed!")
                        return {"success": True, "message": "All quizzes completed"}
                else:
                    logger.warning(f"Quiz {attempt_count} incorrect: {result.get('reason')}")
                    # Check if we got a next URL to skip to
                    next_url = result.get('url')
                    if next_url:
                        logger.info(f"Skipping to next quiz: {next_url}")
                        current_url = next_url
                    else:
                        # No next URL, we're stuck
                        return {"error": f"Failed at quiz {attempt_count}: {result.get('reason')}"}

            except Exception as e:
                logger.exception(f"Error processing quiz {attempt_count}: {e}")
                return {"error": str(e)}

        return {"error": "Max attempts reached or no more quizzes"}

    def solve_single_quiz(self, quiz_url):
        """Solve a single quiz"""
        downloaded_files = []
        try:
            with BrowserManager() as browser:
                # Fetch the quiz page
                logger.info(f"Fetching quiz page: {quiz_url}")
                page_content = browser.fetch_page_content(quiz_url)

                # Extract download links from the page (and normalize relative links)
                download_links = self._extract_download_links(page_content.get('html', ''), base_url=quiz_url)

                # Download any required files
                for link in download_links:
                    try:
                        filename = os.path.basename(urlparse(link).path.split('?')[0])
                        if not filename:
                            filename = f"download_{len(downloaded_files)}.dat"

                        save_path = os.path.join(self.temp_dir, filename)
                        logger.info(f"Downloading: {link} -> {save_path}")
                        browser.download_file(link, save_path)
                        downloaded_files.append(save_path)
                    except Exception as e:
                        logger.exception(f"Error downloading {link}: {e}")

                # Use LLM to solve the quiz
                logger.info("Sending to Claude for solving...")
                solution = self.llm_solver.solve_quiz(page_content, downloaded_files) or {}

                # Submit the answer
                submit_url_raw = solution.get('submit_url')
                answer = solution.get('answer')

                if not submit_url_raw:
                    raise ValueError("No submit URL found in solution")

                # Normalize submit_url relative to the quiz page (handles '/submit', 'submit', './submit', etc.)
                submit_url = self._normalize_url(quiz_url, submit_url_raw)
                logger.info(f"Normalized submit URL: {submit_url}")

                payload = {
                    "email": self.email,
                    "secret": self.secret,
                    "url": quiz_url,
                    "answer": answer
                }

                logger.info(f"Submitting answer to: {submit_url}")
                logger.info(f"Answer: {answer!r}")

                # POST with timeout and robust error handling
                try:
                    response = requests.post(submit_url, json=payload, timeout=30)
                except requests.RequestException as e:
                    logger.exception(f"HTTP error when submitting answer: {e}")
                    return {"correct": False, "reason": f"HTTP error when submitting answer: {e}"}

                if not (200 <= response.status_code < 300):
                    logger.error(f"Non-2xx response from submit endpoint: {response.status_code} {response.text}")
                    return {"correct": False, "reason": f"Submit endpoint returned status {response.status_code}"}

                # Parse JSON safely
                try:
                    result = response.json()
                except ValueError:
                    logger.error("Submit endpoint did not return JSON")
                    return {"correct": False, "reason": "Submit endpoint did not return JSON", "raw_text": response.text}

                logger.info(f"Submission result: {result}")
                return result

        finally:
            # Clean up downloaded files no matter what
            for file_path in downloaded_files:
                try:
                    os.remove(file_path)
                except Exception:
                    logger.debug(f"Could not remove temporary file: {file_path}")

    def _extract_download_links(self, html_content, base_url=None):
        """Extract download links from HTML and normalize them with base_url if relative"""
        # Look for href links to files (absolute or relative)
        pattern = r'href=["\']([^"\']+\.(?:pdf|csv|xlsx?|json|txt|png|jpg|jpeg))["\']'
        raw_links = re.findall(pattern, html_content, re.IGNORECASE)

        normalized = []
        for link in set(raw_links):  # dedupe
            if base_url and not bool(urlparse(link).scheme):
                # relative link -> join with base
                full = urljoin(base_url, link)
            else:
                full = link
            normalized.append(full)

        return normalized

    @staticmethod
    def _normalize_url(base, url):
        """Return absolute URL by joining base and url when needed."""
        if not url:
            return url
        # If url already absolute, urljoin will leave it as-is
        return urljoin(base, url)
