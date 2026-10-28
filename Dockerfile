FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as a non-root user; data/ must be writable for the optional CSV cache
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
