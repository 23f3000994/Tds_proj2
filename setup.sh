#!/bin/bash

echo "Setting up LLM Quiz Solver..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Create necessary directories
mkdir -p temp_downloads
mkdir -p logs

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python app.py"
