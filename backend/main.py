import numpy as np
import tensorflow as tf
import uvicorn
import shutil
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
MODEL_PATH = "rf_fingerprint_model_real.keras"
print(f"Loading classifier model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Classifier loaded successfully.")

ANOMALY_MODEL_PATH = "anomaly_detector.keras"
print(f"Loading anomaly detector from {ANOMALY_MODEL_PATH}...")
anomaly_model = tf.keras.models.load_model(ANOMALY_MODEL_PATH)
print("Anomaly detector loaded successfully.")

with open('device_classes.txt', 'r') as f:
    DEVICE_CLASSES = [line.strip() for line in f.readlines()]
print(f"Loaded {len(DEVICE_CLASSES)} device classes.")

ANOMALY_THRESHOLD = 0.02

app = FastAPI(title="Spectrum Intelligence API")
# --- CORS Middleware ---
allowed_origins_regex = r"https?://rf-fingerprinting-.*\.vercel\.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://rf-fingerprinting.vercel.app"
    ],
    allow_origin_regex=allowed_origins_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Model ---
class PredictionResponse(BaseModel):
    predicted_device: str
    confidence_score: float
    is_anomaly: bool
    details: Dict[str, float]
    signal_magnitude: Optional[List[float]] = None
    reconstruction_error_map: Optional[List[float]] = None

# --- Helper Functions ---
def preprocess_single_signal_array(signal_array: np.ndarray):
    if signal_array.shape[0] < 128:
        pad_width = 128 - signal_array.shape[0]
        signal_array = np.pad(signal_array, (0, pad_width), 'constant', constant_values=(0))
    elif signal_array.shape[0] > 128:
        signal_array = signal_array[:128]
    signal_real_imag = np.stack([signal_array.real, signal_array.imag], axis=-1)
    return np.expand_dims(signal_real_imag, axis=0)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spectrum Intelligence API."}

# --- predict_signal Function with Final Fix ---
@app.post("/predict/", response_model=PredictionResponse)
async def predict_signal(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        signal_data = np.load(temp_file_path, allow_pickle=True)
        if signal_data.ndim == 2: signal_data = signal_data[0, :]
        
        processed_signal_np = preprocess_single_signal_array(signal_data)
        
        reconstruction = anomaly_model.predict(processed_signal_np, verbose=0)
        prediction = model.predict(processed_signal_np, verbose=0)[0]

        overall_reconstruction_error = np.mean(np.square(processed_signal_np - reconstruction))
        is_anomaly = overall_reconstruction_error > ANOMALY_THRESHOLD

        # --- THIS IS THE FINAL, ROBUST FIX FOR THE VISUALIZATION ---
        # 1. Calculate the error for each time step
        error_map = np.mean(np.abs(processed_signal_np - reconstruction), axis=2).flatten()
        
        # 2. Calculate statistics to find a dynamic threshold for "significant" error
        mean_error = np.mean(error_map)
        std_error = np.std(error_map)
        # We define a significant error as anything 2 standard deviations above the mean
        dynamic_threshold = mean_error + (2 * std_error)
        
        # 3. Keep only the errors that are above this threshold, set others to zero
        significant_errors = np.where(error_map > dynamic_threshold, error_map, 0)
        
        # 4. Normalize this new, filtered map to get the final plot values
        max_significant_error = np.max(significant_errors)
        # Add a small epsilon to avoid division by zero if there are no significant errors
        final_error_map = significant_errors / (max_significant_error + 1e-8)
        
        signal_magnitude_processed = np.abs(processed_signal_np[0, :, 0] + 1j * processed_signal_np[0, :, 1]).tolist()

        if is_anomaly:
            predicted_device_name = "Unknown Signal (Anomaly)"
            confidence = float(overall_reconstruction_error)
            details = { "reconstruction_error": confidence }
        else:
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            predicted_device_name = DEVICE_CLASSES[predicted_index]
            details = {DEVICE_CLASSES[j]: float(prediction[j]) for j in range(len(prediction))}
        
        return PredictionResponse(
            predicted_device=predicted_device_name, 
            confidence_score=confidence, 
            is_anomaly=is_anomaly, 
            details=details, 
            signal_magnitude=signal_magnitude_processed,
            reconstruction_error_map=final_error_map.tolist()
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)