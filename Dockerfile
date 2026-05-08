FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/app/input", "/app/output"]

ENV PYTHONUNBUFFERED=1

# Default command runs the indexing pipeline.
# For queries, use:
#   docker-compose run graphrag python -m graphrag.query --root /app --method global "your question"
CMD ["python", "-m", "graphrag.index", "--root", "/app"]
