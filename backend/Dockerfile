FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Опционально: если хотите запускать через uvicorn из контейнера (иначе docker-compose command)
EXPOSE 8000
