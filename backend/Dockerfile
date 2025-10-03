FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y wget && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN wget --no-check-certificate --tries=3 -O RML2016.10a_dict.pkl "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"

RUN python model_trainer.py
RUN python anomaly_trainer.py

RUN python data_generator.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
