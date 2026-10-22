FROM python:3.11.0

RUN apt-get update && apt-get install -y \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kor \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]