#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed successfully"
