import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Student credentials
    STUDENT_EMAIL = os.getenv('STUDENT_EMAIL')
    STUDENT_SECRET = os.getenv('STUDENT_SECRET')
    
    # AIPipe Configuration
    AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
    AIPIPE_MODEL = os.getenv('AIPIPE_MODEL', 'openai/gpt-4o-mini')  # or 'openai/gpt-4.1-nano'
    
    # Server config
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Timeout settings
    QUIZ_TIMEOUT = 180  # 3 minutes in seconds
    BROWSER_TIMEOUT = 30000  # 30 seconds for page loads
    
    # Model settings
    MAX_TOKENS = 4096
