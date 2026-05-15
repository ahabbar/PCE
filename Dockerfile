FROM python:3.11-slim

WORKDIR /app

# System deps for SQLite + scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data logs

# Default to API; override `command` in your platform to run the dashboard:
#   streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.server:app --host :: --port ${PORT}"]
