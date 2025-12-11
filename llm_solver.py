import anthropic
from config import Config
import json
import logging
import base64
import os

logger = logging.getLogger(__name__)

class LLMSolver:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    
    def solve_quiz(self, page_content, downloaded_files=None):
        """Use Claude to solve the quiz"""
        
        # Prepare the message content
        content_parts = [
            {
                "type": "text",
                "text": f"""You are a data analysis expert. You need to solve a quiz question.

Here is the quiz page content:
{page_content['text']}

Instructions:
1. Read the question carefully
2. If files need to be downloaded, they are provided as attachments
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
        
        # Add downloaded files as attachments if any
        if downloaded_files:
            for file_path in downloaded_files:
                try:
                    file_type = self._get_file_type(file_path)
                    
                    if file_type == 'pdf':
                        # For PDFs, send as document
                        with open(file_path, 'rb') as f:
                            pdf_data = base64.standard_b64encode(f.read()).decode('utf-8')
                        content_parts.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data
                            }
                        })
                    elif file_type in ['image']:
                        # For images
                        with open(file_path, 'rb') as f:
                            img_data = base64.standard_b64encode(f.read()).decode('utf-8')
                        media_type = self._get_media_type(file_path)
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data
                            }
                        })
                    else:
                        # For other files, include content as text
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read()
                        content_parts[0]["text"] += f"\n\nFile content from {os.path.basename(file_path)}:\n{file_content[:10000]}"
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        try:
            response = self.client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=Config.MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ]
            )
            
            # Extract the response
            response_text = response.content[0].text
            logger.info(f"Claude response: {response_text}")
            
            # Parse JSON response
            result = json.loads(response_text)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            raise
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
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
