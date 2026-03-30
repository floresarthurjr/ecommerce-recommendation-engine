FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache-friendly layer ordering)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY main.py .
COPY recommend.py .
COPY preprocessing.py .
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]