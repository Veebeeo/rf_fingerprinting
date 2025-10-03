FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown

COPY . .

RUN gdown --id 1aus-u2xSKETW9Yv5Q-QG9tz9Xnbj5yHV

RUN python model_trainer.py
RUN python anomaly_trainer.py

RUN python data_generator.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
