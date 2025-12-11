from playwright.sync_api import sync_playwright
from config import Config
import logging

logger = logging.getLogger(__name__)

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def fetch_page_content(self, url):
        """Fetch and render a JavaScript page"""
        try:
            page = self.browser.new_page()
            page.goto(url, timeout=Config.BROWSER_TIMEOUT, wait_until='networkidle')
            
            # Wait for content to load
            page.wait_for_timeout(2000)
            
            # Get the full HTML content
            html_content = page.content()
            
            # Get visible text
            body_text = page.locator('body').inner_text()
            
            page.close()
            
            return {
                'html': html_content,
                'text': body_text,
                'url': url
            }
        except Exception as e:
            logger.error(f"Error fetching page {url}: {e}")
            raise
    
    def download_file(self, url, save_path):
        """Download a file from URL"""
        try:
            page = self.browser.new_page()
            
            # Start waiting for download
            with page.expect_download() as download_info:
                page.goto(url)
            
            download = download_info.value
            download.save_as(save_path)
            page.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error downloading file {url}: {e}")
            # Fallback to requests
            import requests
            response = requests.get(url)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return save_path
