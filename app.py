from flask import Flask, request, jsonify
from config import Config
from quiz_processor import QuizProcessor
import logging
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "active",
        "message": "LLM Quiz Solver API",
        "endpoints": {
            "/": "GET - This info page",
            "/quiz": "POST - Submit quiz task"
        }
    })

@app.route('/quiz', methods=['POST'])
def handle_quiz():
    """Main endpoint to receive quiz tasks"""
    try:
        # Parse JSON payload
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Verify required fields
        email = data.get('email')
        secret = data.get('secret')
        url = data.get('url')
        
        if not all([email, secret, url]):
            return jsonify({"error": "Missing required fields: email, secret, url"}), 400
        
        # Verify credentials
        if email != Config.STUDENT_EMAIL or secret != Config.STUDENT_SECRET:
            logger.warning(f"Invalid credentials - Email: {email}")
            return jsonify({"error": "Invalid credentials"}), 403
        
        logger.info(f"Received valid quiz request for URL: {url}")
        
        # Process quiz in background to avoid timeout
        def process_async():
            try:
                processor = QuizProcessor(email, secret)
                result = processor.process_quiz_chain(url)
                logger.info(f"Quiz processing completed: {result}")
            except Exception as e:
                logger.error(f"Error in async processing: {e}")
        
        # Start async processing
        thread = threading.Thread(target=process_async)
        thread.start()
        
        # Return immediate response
        return jsonify({
            "status": "accepted",
            "message": "Quiz processing started",
            "url": url
        }), 200
        
    except Exception as e:
        logger.error(f"Error handling quiz request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logger.info(f"Starting Flask app on {Config.HOST}:{Config.PORT}")
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
