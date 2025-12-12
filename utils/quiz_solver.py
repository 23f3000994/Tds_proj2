import json
import logging
import requests
import base64
import re
from typing import Dict, Any, Optional
from urllib.parse import urljoin
from utils.llm_handler import LLMHandler
import time

logger = logging.getLogger(__name__)

class QuizSolver:
    def __init__(self):
        self.llm = LLMHandler()
        self.session = requests.Session()
        
    def solve_quiz(self, browser, quiz_url: str, email: str, secret: str) -> Dict[str, Any]:
        """
        Main method to solve a quiz
        
        Args:
            browser: Browser instance
            quiz_url: URL of the quiz page
            email: Student email
            secret: Student secret
            
        Returns:
            Dictionary with answer or error
        """
        try:
            # Step 1: Get quiz page content
            logger.info(f"Fetching quiz page: {quiz_url}")
            html_content = browser.get_page_content(quiz_url)
            
            if not html_content:
                return {"error": "Failed to load quiz page"}
            
            # Step 2: Analyze the quiz page
            analysis = self.llm.analyze_quiz_page(html_content)
            logger.info(f"Quiz analysis: {analysis.get('question', 'Unknown')}")
            
            # Step 3: Extract instructions from HTML (fallback method)
            instructions = self._extract_instructions(html_content)
            
            # Step 4: Execute the required tasks
            answer = self._execute_tasks(analysis, instructions, browser)
            
            # Step 5: Prepare submission
            submit_url = analysis.get('submit_url') or self._extract_submit_url(html_content)
            
            if not submit_url:
                return {"error": "No submission URL found"}
            
            # Step 6: Submit answer
            submission_result = self._submit_answer(
                submit_url, email, secret, quiz_url, answer
            )
            
            return submission_result
            
        except Exception as e:
            logger.error(f"Error solving quiz: {str(e)}")
            return {"error": str(e)}
    
    def _extract_instructions(self, html_content: str) -> Dict[str, Any]:
        """Extract instructions from HTML content"""
        instructions = {}
        
        # Look for base64 encoded instructions (common in the sample)
        base64_pattern = r'atob\(["\']([A-Za-z0-9+/=]+)["\']'
        matches = re.findall(base64_pattern, html_content)
        
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8')
                instructions['decoded_content'] = decoded
                
                # Extract URLs
                url_pattern = r'https?://[^\s<>"\']+'
                urls = re.findall(url_pattern, decoded)
                if urls:
                    instructions['urls'] = urls
                    
                # Extract JSON
                json_pattern = r'\{[^{}]*\}'
                json_matches = re.findall(json_pattern, decoded, re.DOTALL)
                if json_matches:
                    instructions['json_blocks'] = json_matches
                    
            except:
                continue
        
        return instructions
    
    def _extract_submit_url(self, html_content: str) -> Optional[str]:
        """Extract submission URL from HTML"""
        # Look for common patterns
        patterns = [
            r'Post your answer to (https?://[^\s]+)',
            r'submit.*?(https?://[^\s]+)',
            r'action=["\'](https?://[^\s"\'<>]+)["\']',
            r'https?://[^\s]+/submit',
            r'https?://[^\s]+/answer',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def _execute_tasks(self, analysis: Dict[str, Any], instructions: Dict[str, Any], browser) -> Any:
        """Execute the required tasks based on analysis"""
        try:
            # Extract task type from analysis
            question = analysis.get('question', '').lower()
            
            # Handle different types of tasks
            if any(word in question for word in ['download', 'file', 'pdf']):
                return self._handle_file_task(analysis, instructions, browser)
            elif any(word in question for word in ['sum', 'calculate', 'total', 'average']):
                return self._handle_calculation_task(analysis, instructions, browser)
            elif any(word in question for word in ['scrape', 'extract', 'find']):
                return self._handle_scraping_task(analysis, instructions, browser)
            else:
                # Generic handling using LLM
                return self._handle_generic_task(analysis, instructions)
                
        except Exception as e:
            logger.error(f"Error executing tasks: {str(e)}")
            return None
    
    def _handle_file_task(self, analysis: Dict[str, Any], instructions: Dict[str, Any], browser) -> Any:
        """Handle tasks involving file downloads"""
        # Extract file URL
        file_urls = []
        
        if 'urls' in instructions:
            file_urls = [url for url in instructions['urls'] if any(ext in url for ext in ['.pdf', '.csv', '.xlsx', '.json'])]
        
        if not file_urls and 'data_sources' in analysis:
            file_urls = analysis['data_sources']
        
        if not file_urls:
            # Try to find download links in HTML
            # This would need actual browser interaction
            pass
        
        # Download and process files
        results = []
        for file_url in file_urls[:3]:  # Limit to 3 files
            try:
                # Download file
                response = self.session.get(file_url, timeout=30)
                
                # Process based on file type
                if file_url.endswith('.csv'):
                    import pandas as pd
                    import io
                    df = pd.read_csv(io.StringIO(response.text))
                    results.append(df)
                elif file_url.endswith('.json'):
                    results.append(response.json())
                else:
                    # For other file types, store content
                    results.append(response.content[:1000])  # First 1000 bytes
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_url}: {str(e)}")
        
        # Use LLM to analyze the question and data
        task_description = analysis.get('question', '')
        combined_data = str(results)
        
        return self.llm.solve_data_task(combined_data, task_description)
    
    def _handle_calculation_task(self, analysis: Dict[str, Any], instructions: Dict[str, Any], browser) -> Any:
        """Handle calculation tasks"""
        # This would involve downloading data and performing calculations
        # For now, use LLM with available data
        task_description = analysis.get('question', '')
        
        # Try to extract data from instructions
        data = instructions.get('decoded_content', '') or str(instructions)
        
        return self.llm.solve_data_task(data, task_description)
    
    def _handle_scraping_task(self, analysis: Dict[str, Any], instructions: Dict[str, Any], browser) -> Any:
        """Handle web scraping tasks"""
        # Extract URLs to scrape
        urls_to_scrape = []
        
        if 'urls' in instructions:
            urls_to_scrape = instructions['urls']
        
        # Scrape each URL
        scraped_data = []
        for url in urls_to_scrape[:5]:  # Limit to 5 URLs
            try:
                content = browser.get_page_content(url)
                if content:
                    # Extract text content (simplified)
                    text_content = re.sub('<[^<]+?>', ' ', content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    scraped_data.append(f"URL: {url}\nContent: {text_content[:1000]}")
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {str(e)}")
        
        # Analyze with LLM
        task_description = analysis.get('question', '')
        combined_data = '\n\n'.join(scraped_data)
        
        return self.llm.solve_data_task(combined_data, task_description)
    
    def _handle_generic_task(self, analysis: Dict[str, Any], instructions: Dict[str, Any]) -> Any:
        """Handle generic tasks using LLM"""
        task_description = analysis.get('question', '')
        available_data = {
            'analysis': analysis,
            'instructions': instructions
        }
        
        return self.llm.solve_data_task(str(available_data), task_description)
    
    def _submit_answer(self, submit_url: str, email: str, secret: str, 
                      quiz_url: str, answer: Any) -> Dict[str, Any]:
        """Submit answer to the specified URL"""
        try:
            # Prepare payload
            payload = {
                "email": email,
                "secret": secret,
                "url": quiz_url,
                "answer": answer
            }
            
            # Submit
            response = self.session.post(
                submit_url,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            logger.info(f"Submission result: {result.get('correct', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to submit answer: {str(e)}")
            return {"error": f"Submission failed: {str(e)}", "correct": False}
