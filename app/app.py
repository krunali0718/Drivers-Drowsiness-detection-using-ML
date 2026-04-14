import streamlit as st
import cv2
import sys
import os
import time

# Append root directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection import get_face_landmarks, crop_face_for_cnn
from src.features import extract_features
from src.predict import current_model, predict_cnn_state, calculate_system_fatigue

st.set_page_config(page_title="Vision AI Driver Safety", layout="wide")

st.title("🚗 Deep Learning Driver Drowsiness System")
st.markdown("""
Welcome to the Next-Gen **Vision AI Platform**. 
This application utilizes a custom 99% accuracy Deep Convolutional Neural Network (CNN) combined with MediaPipe 3D precise facial geometry to analyze driver fatigue instantly without false hardware alarms.
""")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Live Feed Analysis")
    run_camera = st.checkbox("Toggle Camera Stream", value=False)
    FRAME_WINDOW = st.image([])

with col2:
    st.markdown("### 🧠 Live Diagnostics")
    state_placeholder = st.empty()
    fatigue_placeholder = st.empty()
    ear_placeholder = st.empty()
    st.markdown("---")
    st.info("💡 **Viva Concept:**\n\nInput (Webcam) → Processing (MediaPipe Isolation) → Model (Deep CNN Vision) → Output (Live Inference)")

# State tracking
last_log = time.time()

if run_camera:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. PROCESSING (MediaPipe Isolation)
        face_landmarks = get_face_landmarks(rgb_frame)
        
        # Display Defaults
        color = (0, 255, 0)
        display_state = "Awake 😊"
        fatigue_score = 0.0
        ear_val = 0.0
        
        if face_landmarks:
            for face in face_landmarks:
                # 2. FEATURE EXTRACTION
                features = extract_features(face, w, h)
                ear_val = features["EAR"]
                
                # Plot the bounding box using detection logic
                face_crop, box = crop_face_for_cnn(frame, face, w, h)
                if box is not None:
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 0), 2)
                    cv2.putText(frame, "CNN ROI", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                
                # 3. MODEL (Deep CNN + Fatigue Fusion)
                cnn_pred, confidence, cnn_modifier = predict_cnn_state(current_model, face_crop)
                fatigue_score = calculate_system_fatigue(features, cnn_modifier)
                
                # 4. OUTPUT / ALERT LOGIC
                if fatigue_score > 40.0:
                    display_state = "Drowsy 😴"
                    color = (0, 0, 255)
                    cv2.putText(frame, "WARNING: CRITICAL FATIGUE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                elif cnn_pred == "Drowsy":
                    display_state = "Drowsy (Micro-Sleep) 😴"
                    color = (0, 165, 255)
                    cv2.putText(frame, "ALERT: MICRO-SLEEP DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    display_state = "Awake 😊"
                    color = (0, 255, 0)
                    cv2.putText(frame, "Driver is Focused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Render the Annotated Frame on Streamlit Dashboard
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        
        # Render Analytics
        if display_state == "Awake 😊":
            state_placeholder.success(f"**Status:** {display_state}")
        else:
            state_placeholder.error(f"**Status:** {display_state}")
            
        fatigue_placeholder.metric(label="Fatigue Score", value=f"{fatigue_score:.1f}%")
        ear_placeholder.metric(label="Eye Aspect Ratio (EAR)", value=f"{ear_val:.3f}")
        
        # Required to flush streamlit updates effectively
        time.sleep(0.01)
        
    cap.release()
else:
    st.warning("Camera is OFF. Please toggle the checkbox to start Real-Time Analysis.")
