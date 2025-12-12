from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
import time

logger = logging.getLogger(__name__)

class BrowserManager:
    def __init__(self):
        self.driver = None
        self.options = Options()
        
        # Configure Chrome options
        self._setup_options()
    
    def _setup_options(self):
        """Setup Chrome browser options"""
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--window-size=1920,1080')
        
        # Add headless option if configured
        from config import Config
        config = Config()
        if config.HEADLESS:
            self.options.add_argument('--headless')
        
        # Additional options for better performance
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
    
    def start_browser(self):
        """Start the browser session"""
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.options)
            
            # Execute CDP commands to prevent detection
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Browser started successfully")
            return self.driver
        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}")
            raise
    
    def get_browser(self):
        """Get or create browser instance"""
        if not self.driver:
            return self.start_browser()
        return self.driver
    
    def get_page_content(self, url, wait_time=10):
        """Navigate to URL and get page content"""
        try:
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for JavaScript execution
            time.sleep(2)
            
            # Get page source
            page_source = self.driver.page_source
            
            # Execute any JavaScript to get dynamically rendered content
            rendered_content = self.driver.execute_script("""
                return document.documentElement.outerHTML;
            """)
            
            return rendered_content or page_source
            
        except Exception as e:
            logger.error(f"Error loading page {url}: {str(e)}")
            return None
    
    def close_browser(self):
        """Close the browser session"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser closed")
