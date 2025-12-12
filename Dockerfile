FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=10000

CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
