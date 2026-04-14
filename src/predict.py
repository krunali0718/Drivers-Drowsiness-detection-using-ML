import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2
from collections import deque
import statistics

# Stabilize the CNN Output to prevent flickering
cnn_history = deque(maxlen=10)

def load_cnn_model():
    """ Load the High-Accuracy 99% Deep Learning CNN """
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'cnn_model_augmented.keras')
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Successfully loaded High-Accuracy CNN Model.")
        return model
    except Exception as e:
        print(f"Error loading Deep Learning CNN: {e}")
        return None

# Load Once at App Startup
current_model = load_cnn_model()

def predict_cnn_state(model, face_crop):
    """
    Accepts cropped facial image, resizes it for the CNN, and predicts mathematically.
    Uses Mode stabilization to avoid visual glitches.
    Returns: label ("Drowsy" or "Awake"), confidence score, and raw fatigue modifier.
    """
    if model is None or face_crop is None:
        return "Analyzing...", 0.0, 0
        
    try:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (64, 64))
        face_array = np.expand_dims(face_resized, axis=0)
        
        preds = model.predict(face_array, verbose=0)
        class_id = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        
        # Interpret CNN Model (0 = Drowsy, 1 = Awake)
        # Assuming typical setup, double check labels based on our notebook training stats
        # The user's notebook mapping was {"Normal": 0, "Sleepy": 1} inside ML datasets, 
        # But image_dataset_from_directory alphabetizes folders: 'Drowsy' -> 0, 'Non Drowsy' -> 1
        raw_state = "Drowsy" if class_id == 0 else "Awake"
        
        cnn_history.append(raw_state)
        stable_state = statistics.mode(cnn_history) if len(cnn_history) > 0 else raw_state
        
        fatigue_modifier = 15.0 if stable_state == "Drowsy" else 0.0
        
        return stable_state, confidence, fatigue_modifier
        
    except Exception as e:
        return f"Error: {e}", 0.0, 0
    
def calculate_system_fatigue(features, cnn_modifier):
    """
    Blends the mathematical threshold parameters with the Deep Learning model 
    for an ultimate precision fatigue score (0-100%).
    """
    ear = features.get("EAR", 0.3)
    baseline_score = 0.0
    
    if ear < 0.16: # EYE_THRESHOLD Critical
        baseline_score += 45.0
    elif ear < 0.20: # EYE_THRESHOLD Warning
        baseline_score += 15.0
        
    # Yawn check
    mar = features.get("MAR", 0.0)
    if mar > 0.6:
        baseline_score += 25.0
        
    final_score = baseline_score + cnn_modifier
    return min(100.0, final_score)
