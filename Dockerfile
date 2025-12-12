FROM python:3.11-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxss1 \
    libglib2.0-0 \
    libglib2.0-bin \
    libxkbcommon0 \
    libxfixes3 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright Python package
RUN pip install playwright

# IMPORTANT: Install Chromium browser inside the image
RUN playwright install chromium

WORKDIR /app

COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render
ENV PORT=10000

CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
