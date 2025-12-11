import requests
from config import Config
import json
import logging
import base64
import os

logger = logging.getLogger(__name__)

class LLMSolver:
    def __init__(self):
        self.api_url = "https://aipipe.org/openrouter/v1/chat/completions"
        self.token = Config.AIPIPE_TOKEN
    
    def solve_quiz(self, page_content, downloaded_files=None):
        """Use AIPipe (GPT-4o-mini/nano) to solve the quiz"""
        
        # Prepare the message content
        messages = [
            {
                "role": "system",
                "content": "You are a helpful data analysis assistant. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": f"""You are a data analysis expert. You need to solve a quiz question.

Here is the quiz page content:
{page_content['text']}

Instructions:
1. Read the question carefully
2. If files need to be downloaded, they are provided below
3. Perform any required data analysis, visualization, or processing
4. Extract the submit URL and required payload format from the question
5. Return your answer in this JSON format:
{{
    "submit_url": "the URL to submit to",
    "answer": your_answer_here,
    "reasoning": "brief explanation of your solution"
}}

The answer can be a number, string, boolean, object, or base64-encoded data URI.
Make sure to follow the exact format requested in the question.

IMPORTANT: Only respond with valid JSON. No extra text before or after."""
            }
        ]
        
        # Add downloaded files content
        if downloaded_files:
            for file_path in downloaded_files:
                try:
                    file_type = self._get_file_type(file_path)
                    
                    if file_type == 'image':
                        # For images, we'll include them as base64 in the message
                        with open(file_path, 'rb') as f:
                            img_data = base64.standard_b64encode(f.read()).decode('utf-8')
                        media_type = self._get_media_type(file_path)
                        
                        messages[1]["content"] += f"\n\n[Image file: {os.path.basename(file_path)} - base64 data included]"
                        # Note: Some models might not support image input via AIPipe
                        # In that case, describe the image or use OCR
                        
                    else:
                        # For text/data files, include content directly
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read()
                        messages[1]["content"] += f"\n\nFile content from {os.path.basename(file_path)}:\n{file_content[:20000]}"
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        try:
            # Make API call to AIPipe
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": Config.AIPIPE_MODEL,  # e.g., "openai/gpt-4o-mini"
                "messages": messages,
                "max_tokens": Config.MAX_TOKENS,
                "temperature": 0.1
            }
            
            logger.info(f"Calling AIPipe with model: {Config.AIPIPE_MODEL}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            response_text = result['choices'][0]['message']['content']
            logger.info(f"AIPipe response: {response_text}")
            
            # Clean response if it has markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            parsed_result = json.loads(response_text)
            return parsed_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling AIPipe API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def _get_file_type(self, file_path):
        """Determine file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            return 'image'
        elif ext in ['.csv', '.txt', '.json', '.xml']:
            return 'text'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        return 'unknown'
    
    def _get_media_type(self, file_path):
        """Get media type for image files"""
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mapping.get(ext, 'image/jpeg')
