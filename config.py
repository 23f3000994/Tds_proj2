import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Student credentials
    STUDENT_EMAIL = os.getenv('STUDENT_EMAIL')
    STUDENT_SECRET = os.getenv('STUDENT_SECRET')
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Server config
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Timeout settings
    QUIZ_TIMEOUT = 180  # 3 minutes in seconds
    BROWSER_TIMEOUT = 30000  # 30 seconds for page loads
    
    # Claude model
    CLAUDE_MODEL = 'claude-sonnet-4-20250514'
    MAX_TOKENS = 4096
