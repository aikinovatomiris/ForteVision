FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY ml ./ml
COPY scripts ./scripts
COPY configs ./configs
RUN useradd --create-home worker && mkdir -p /app/.uploads /app/models && chown -R worker:worker /app
USER worker
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
