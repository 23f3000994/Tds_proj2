import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Student configuration (from Google Form)
    EMAIL = os.getenv('EMAIL', 'your-email@example.com')
    SECRET = os.getenv('SECRET', 'your-secret-string')
    
    # LLM Configuration (using AIPipe)
    AIPIPE_API_KEY = os.getenv('AIPIPE_API_KEY', 'your-aipipe-api-key')
    AIPIPE_BASE_URL = os.getenv('AIPIPE_BASE_URL', 'https://api.aipipe.org/v1')
    
    # Browser Configuration
    HEADLESS = os.getenv('HEADLESS', 'true').lower() == 'true'
    BROWSER_TIMEOUT = int(os.getenv('BROWSER_TIMEOUT', 30000))  # 30 seconds
    
    # Quiz Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    TIMEOUT_SECONDS = int(os.getenv('TIMEOUT_SECONDS', 180))  # 3 minutes
    
    # System and User Prompts (from Google Form)
    SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful assistant.')
    USER_PROMPT = os.getenv('USER_PROMPT', 'Please help me with this task.')
