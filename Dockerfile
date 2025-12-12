FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libasound2 libx11-6 libx11-xcb1 libxcb1 libxss1 \
    libglib2.0-0 libglib2.0-bin libxkbcommon0 libxfixes3 libpango-1.0-0 \
    libcairo2 wget \
 && rm -rf /var/lib/apt/lists/*

# Install Playwright + browser deps
RUN pip install playwright && playwright install chromium

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
