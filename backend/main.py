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

# --- OPTIMIZED FUNCTIONS ---

@tf.function
def predict_and_get_saliency(input_tensor):
    """
    OPTIMIZED to perform prediction, find the top class, and get saliency in one pass.
    """
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor, training=False)
        # Find the top prediction and its index *inside* the compiled graph
        predicted_index = tf.argmax(predictions[0])
        top_prediction = predictions[0, predicted_index]
    
    grads = tape.gradient(top_prediction, input_tensor)
    # Return everything we need
    return predictions, grads, predicted_index

@tf.function
def anomaly_and_get_saliency(input_tensor):
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        reconstruction = anomaly_model(input_tensor, training=False)
        loss = tf.reduce_mean(tf.square(input_tensor - reconstruction))

    grads = tape.gradient(loss, input_tensor)
    return reconstruction, grads


app = FastAPI(title="Spectrum Intelligence API")

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

class PredictionResponse(BaseModel):
    predicted_device: str
    confidence_score: float
    is_anomaly: bool
    details: Dict[str, float]
    saliency_map: Optional[List[float]] = None
    signal_magnitude: Optional[List[float]] = None
    demodulated_data: Optional[str] = None

def preprocess_single_signal_array(signal_array: np.ndarray):
    if signal_array.shape[0] < 128:
        pad_width = 128 - signal_array.shape[0]
        signal_array = np.pad(signal_array, (0, pad_width), 'constant', constant_values=(0))
    elif signal_array.shape[0] > 128:
        signal_array = signal_array[:128]
    signal_real_imag = np.stack([signal_array.real, signal_array.imag], axis=-1)
    return np.expand_dims(signal_real_imag, axis=0)

def demodulate_qpsk(signal_data: np.ndarray) -> str:
    bits = ""
    symbols = signal_data[::2]
    for symbol in symbols:
        if symbol.real > 0 and symbol.imag > 0: bits += "11"
        elif symbol.real < 0 and symbol.imag > 0: bits += "01"
        elif symbol.real < 0 and symbol.imag < 0: bits += "00"
        elif symbol.real > 0 and symbol.imag < 0: bits += "10"
    return bits

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spectrum Intelligence API."}

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
        signal_magnitude = np.abs(signal_data).tolist()
        
        processed_signal_tf = tf.convert_to_tensor(processed_signal_np, dtype=tf.float32)

        reconstruction, anomaly_grads = anomaly_and_get_saliency(processed_signal_tf)
        reconstruction_error = np.mean(np.square(processed_signal_np - reconstruction.numpy()))
        is_anomaly = reconstruction_error > ANOMALY_THRESHOLD
        
        if is_anomaly:
            saliency = np.mean(np.abs(anomaly_grads.numpy()), axis=-1).flatten()
            saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
            return PredictionResponse(predicted_device="Unknown Signal (Anomaly)", confidence_score=float(reconstruction_error), is_anomaly=True, details={"reconstruction_error": float(reconstruction_error)}, saliency_map=saliency.tolist(), signal_magnitude=signal_magnitude)
            
        # --- REMOVED THE REDUNDANT PREDICTION CALL ---
        # The optimized function now returns everything we need in one go.
        prediction_tf, grads_tf, predicted_index_tf = predict_and_get_saliency(processed_signal_tf)
        
        # Convert tensors to numpy arrays for post-processing
        prediction = prediction_tf.numpy()[0]
        predicted_index = predicted_index_tf.numpy()
        grads = grads_tf.numpy()
        
        confidence = float(np.max(prediction))
        predicted_device_name = DEVICE_CLASSES[predicted_index]
        details = {DEVICE_CLASSES[j]: float(prediction[j]) for j in range(len(prediction))}
        
        saliency = np.mean(np.abs(grads), axis=-1).flatten()
        saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
        
        demodulated_data = None
        if predicted_device_name == "QPSK": demodulated_data = demodulate_qpsk(signal_data)
        
        return PredictionResponse(predicted_device=predicted_device_name, confidence_score=confidence, is_anomaly=False, details=details, saliency_map=saliency.tolist(), signal_magnitude=signal_magnitude, demodulated_data=demodulated_data)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)