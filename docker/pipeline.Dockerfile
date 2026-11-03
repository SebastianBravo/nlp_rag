FROM python:3.12-slim
WORKDIR /work
COPY requirements/torch-cpu.txt requirements/api.txt requirements/pipeline.txt requirements/
RUN pip install --no-cache-dir -r requirements/torch-cpu.txt && pip install --no-cache-dir -r requirements/pipeline.txt
# Modelos de oraciones de nltk (punkt) para la segmentación por estructura (Fase B).
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab
COPY pytest.ini .
COPY app/ app/
COPY pipeline/ pipeline/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
ENV PYTHONUNBUFFERED=1
CMD ["pytest", "-q"]
