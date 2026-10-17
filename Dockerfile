FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY ml ./ml
COPY scripts ./scripts
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN useradd --uid 10001 --create-home worker && mkdir -p /app/models /var/lib/fortevision /var/tmp/fortevision && chown -R worker:worker /app /var/lib/fortevision /var/tmp/fortevision
USER worker
EXPOSE 8000 8501
CMD ["python", "-m", "scripts.serve", "--host", "0.0.0.0"]
