FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

RUN adduser --disabled-password --gecos "" --uid 10001 airman

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image, then run fully offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && chown -R airman:airman /opt/hf

COPY app/ ./app/
COPY web/ ./web/
COPY vectorstore/ ./vectorstore/
COPY evaluate.py evaluation_set.json out_of_scope_set.json ./

ENV HF_LOCAL_FILES_ONLY=1
USER airman
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if b['status']=='ok' else 1)"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
