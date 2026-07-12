FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 supervisor && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /var/log/supervisor

COPY API_App/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8080

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
