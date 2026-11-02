FROM python:3.12-slim
WORKDIR /work
COPY requirements/torch-cpu.txt requirements/api.txt requirements/
RUN pip install --no-cache-dir -r requirements/torch-cpu.txt && pip install --no-cache-dir -r requirements/api.txt
COPY app/ app/
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
