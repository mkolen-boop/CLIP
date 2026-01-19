FROM python:3.11-slim

# Системні залежності для Pillow та базових операцій з зображеннями
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Встановлюємо залежності
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копіюємо код
COPY app /app/app

# Запуск сервера
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
