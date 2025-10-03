FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y wget bzip2 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN wget -O RML2016.10a.tar.bz2 "https://s3.amazonaws.com/redwood-open-data/radio_ml_2016/RML2016.10a.tar.bz2" && \
    tar -xjf RML2016.10a.tar.bz2 && \
    mv RML2016.10a_dict.pkl . && \
    rm RML2016.10a.tar.bz2

RUN python model_trainer.py
RUN python anomaly_trainer.py
RUN python data_generator.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
