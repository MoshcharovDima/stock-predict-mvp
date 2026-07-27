FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FINNHUB_API_KEY передаётся через -e при запуске
EXPOSE 8501
# healthcheck на python: curl в python:3.11-slim отсутствует
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; \
    urllib.request.urlopen('http://localhost:8501/_stcore/health').read()" \
    || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
