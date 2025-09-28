import numpy as np
import tensorflow as tf
import uvicorn
import shutil
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict


MODEL_PATH = "rf_fingerprint_model.keras"
DEVICE_CLASSES = ["friendly_drone", "hostile_drone", "iot_device"] 

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

app = FastAPI(title="AETHER RF Sentinel API")

class PredictionResponse(BaseModel):
    predicted_device: str
    confidence_score: float
    details: Dict[str, float]

def preprocess_single_signal(file_path):
    """
    Loads a single audio file and preprocesses it just like the training data.
    """
 
    signal = np.load(file_path)
    
   
    if len(signal) != 1024:

        raise ValueError(f"Signal length is {len(signal)}, expected 1024.")

    
    signal_real_imag = np.stack([signal.real, signal.imag], axis=-1)
 
    return np.expand_dims(signal_real_imag, axis=0)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AETHER RF Sentinel API. Use the /predict endpoint to classify signals."}

@app.post("/predict/", response_model=PredictionResponse)
async def predict_signal(file: UploadFile = File(...)):
    """
    Accepts a .npy signal file, classifies it, and returns the prediction.
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

     
        processed_signal = preprocess_single_signal(temp_file_path)

      
        prediction = model.predict(processed_signal)[0]

        
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_device_name = DEVICE_CLASSES[predicted_index]

       
        details = {DEVICE_CLASSES[i]: float(prediction[i]) for i in range(len(prediction))}

        return {
            "predicted_device": predicted_device_name,
            "confidence_score": confidence,
            "details": details
        }
    finally:
   
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

