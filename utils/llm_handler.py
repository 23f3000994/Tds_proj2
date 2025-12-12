import requests
import json
import logging
from typing import Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.config = Config()
        self.api_key = self.config.AIPIPE_API_KEY
        self.base_url = self.config.AIPIPE_BASE_URL
        
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4") -> str:
        """
        Call LLM via AIPipe API
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            model: Model to use
            
        Returns:
            LLM response as string
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {str(e)}")
            raise
    
    def analyze_quiz_page(self, html_content: str, task_description: str = "") -> Dict[str, Any]:
        """
        Analyze quiz page HTML and extract instructions
        
        Args:
            html_content: HTML content of the quiz page
            task_description: Additional task description
            
        Returns:
            Dictionary with analysis results
        """
        prompt = f"""
        Analyze this quiz page HTML and extract the following information:
        
        1. What is the quiz question/task?
        2. What data/files need to be processed?
        3. What is the expected output format?
        4. Where should the answer be submitted (URL)?
        5. What is the submission format (JSON structure)?
        
        HTML Content:
        {html_content[:10000]}  # Limit to first 10k chars
        
        Additional context: {task_description}
        
        Provide your analysis in JSON format with these keys:
        - question: The main question/task
        - data_sources: List of data sources/files mentioned
        - expected_output: Description of expected output
        - submit_url: URL for submission
        - submission_format: JSON structure for submission
        - steps_needed: List of steps required to solve
        """
        
        response = self.call_llm(prompt, system_prompt="You are a quiz analysis expert.")
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            return json.loads(json_str)
        except:
            # Fallback if JSON parsing fails
            return {
                "question": "Unable to parse",
                "raw_response": response
            }
    
    def solve_data_task(self, data: Any, task: str) -> Any:
        """
        Solve data analysis task using LLM
        
        Args:
            data: Data to analyze (could be text, structured data, etc.)
            task: Description of what to do with the data
            
        Returns:
            Analysis result
        """
        prompt = f"""
        Task: {task}
        
        Data: {str(data)[:5000]}  # Limit data size
        
        Instructions:
        1. Analyze the data carefully
        2. Perform the requested operation
        3. Provide only the final answer in the required format
        4. If calculation is needed, show your work but put final answer at the end
        
        Your response should be ONLY the answer in the appropriate format.
        """
        
        return self.call_llm(prompt, system_prompt="You are a data analysis expert.")
