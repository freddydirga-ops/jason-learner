FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jason_learn.py .

RUN mkdir -p /data

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

CMD ["python3", "jason_learn.py", "run"]
