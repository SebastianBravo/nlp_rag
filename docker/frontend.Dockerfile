FROM python:3.12-slim
WORKDIR /work
COPY requirements/frontend.txt requirements/
RUN pip install --no-cache-dir -r requirements/frontend.txt
COPY app/frontend.py app/
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "app/frontend.py", "--server.address=0.0.0.0", "--server.port=8501"]
