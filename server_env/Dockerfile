# Используем легковесный образ с Python
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей для работы нейросетевых библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python пакеты
COPY requirements.txt .
# Устанавливаем tensorflow-cpu, чтобы не тянуть гигабайты CUDA библиотек на сервер (инференс на CPU идет быстро)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем веб-сервер
COPY app.py .
COPY templates/ templates/
COPY static/ static/
EXPOSE 5000

# Для Production используем gunicorn вместо встроенного dev-сервера Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
