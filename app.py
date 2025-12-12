import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import requests
from config import Config
from utils.quiz_solver import QuizSolver
from utils.browser import BrowserManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
quiz_solver = QuizSolver()
browser_manager = BrowserManager()

# Store active quiz sessions
active_sessions = {}

class APIServer:
    def __init__(self):
        self.config = Config()
        
    def verify_secret(self, email, secret):
        """Verify if the secret matches for the given email"""
        # In production, this would check against a database
        # For now, using config
        return secret == self.config.SECRET
    
    def process_quiz(self, email, secret, quiz_url):
        """Process quiz URL and return answer"""
        try:
            # Start browser session
            browser = browser_manager.get_browser()
            
            # Solve the quiz
            result = quiz_solver.solve_quiz(browser, quiz_url, email, secret)
            
            return result
        except Exception as e:
            logger.error(f"Error processing quiz: {str(e)}")
            raise

api_server = APIServer()

@app.route('/quiz', methods=['POST'])
def handle_quiz():
    """Handle quiz POST requests"""
    try:
        # Parse JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Extract required fields
        email = data.get('email')
        secret = data.get('secret')
        quiz_url = data.get('url')
        
        # Validate required fields
        if not all([email, secret, quiz_url]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Verify secret
        if not api_server.verify_secret(email, secret):
            return jsonify({"error": "Invalid secret"}), 403
        
        # Process the quiz
        logger.info(f"Processing quiz for {email}: {quiz_url}")
        
        # Start processing in background (async would be better in production)
        # For simplicity, processing synchronously
        result = api_server.process_quiz(email, secret, quiz_url)
        
        return jsonify(result), 200
        
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()}), 200

@app.route('/demo', methods=['POST'])
def demo_endpoint():
    """Demo endpoint for testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Simple demo response
        return jsonify({
            "message": "Demo endpoint working",
            "received_data": data,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Start browser on startup
    browser_manager.start_browser()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
