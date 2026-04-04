import cv2
import mediapipe as mp
import time
import math
import winsound
import joblib
import pandas as pd
import sqlite3
import os
from collections import deque
from datetime import datetime
import statistics

# ==========================================
# ADVANCED DRIVER FATIGUE SYSTEM V3
# Multi-Condition AI with XGBoost + CNN-Level Features + SQLite DB Dashboard
# ==========================================

# 1. Initialize SQLite Local Database (Professional System)
DB_FILE = "driver_fatigue_database.sqlite"

# Connect to database (it creates it if it doesn't exist)
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

# Create standard SQL Table for storing event logs
cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        alert_type TEXT,
        severity TEXT,
        perclos_value REAL
    )
''')
conn.commit()

def log_to_database(alert_type, severity, perclos_val):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Using SQL to securely insert data into our Relational Database
    cursor.execute('''
        INSERT INTO alerts (timestamp, alert_type, severity, perclos_value)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, alert_type, severity, round(perclos_val, 2)))
    conn.commit()

# 2. Load the Advanced XGBoost Model and Scaler
try:
    model = joblib.load("xgboost_drowsiness_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    print("Successfully loaded Advanced XGBoost Model and Scaler.")
except Exception as e:
    print(f"Warning: Could not load advanced XGBoost model. Make sure to run 'advanced_train.py' first.")
    print(f"Error: {e}")
    # Fallback to old model structure just in case
    try:
        model = joblib.load("drowsiness_model.pkl")
        scaler = None
    except:
        model = None
        scaler = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [13,14,78,308]
NOSE = 1; LEFT_FACE = 234; RIGHT_FACE = 454; CHIN = 152

def distance(p1,p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

EYE_THRESHOLD = 0.16 
ear_history = deque(maxlen=150) 
head_history = deque(maxlen=60) 
ml_history = deque(maxlen=10)   
stress_history = deque(maxlen=60) 
yawn_history = deque(maxlen=100)

FONT_STYLE = cv2.FONT_HERSHEY_COMPLEX 
ALERT_SCALE = 0.70
DASH_SCALE = 0.45

EYE_CLOSE_START = None
DROWSY_TIME = 1.5 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Advanced Drowsiness Detection System", cv2.WINDOW_NORMAL)

last_log_time = time.time()
night_mode = False

print("System Ready! Press 'n' to toggle Night Mode. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)

    # --- NIGHT MODE ENHANCEMENT ---
    if night_mode:
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    phone_detected = False
    head_dir = "FORWARD"
    ml_pred = "Analyzing..."
    perclos = 0.0
    is_stressed = False
    fatigue_score = 0.0 

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # --- DISTRACTION / POSTURE ---
            nose = face_landmarks.landmark[NOSE]; left_face = face_landmarks.landmark[LEFT_FACE]
            right_face = face_landmarks.landmark[RIGHT_FACE]; chin = face_landmarks.landmark[CHIN]
            nose_x = int(nose.x * w); left_x = int(left_face.x * w); right_x = int(right_face.x * w); chin_y = int(chin.y * h)
            face_center = (left_x + right_x) // 2

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    if wrist_y < chin_y + 20: phone_detected = True

            tilt_angle = 0
            if phone_detected: head_dir = "ON PHONE CALL! (DISTRACTED)"
            elif nose_x < face_center - 40: tilt_angle = -20; head_dir = "LOOKING LEFT (DISTRACTED)"
            elif nose_x > face_center + 40: tilt_angle = 20; head_dir = "LOOKING RIGHT (DISTRACTED)"
            elif int(nose.y*h) > chin_y - 25: tilt_angle = 25; head_dir = "LOOKING DOWN (DISTRACTED)"

            head_history.append(head_dir)

            # --- EAR ---
            left_ear = distance(face_landmarks.landmark[LEFT_EYE[1]], face_landmarks.landmark[LEFT_EYE[5]]) / distance(face_landmarks.landmark[LEFT_EYE[0]], face_landmarks.landmark[LEFT_EYE[3]])
            right_ear = distance(face_landmarks.landmark[RIGHT_EYE[1]], face_landmarks.landmark[RIGHT_EYE[5]]) / distance(face_landmarks.landmark[RIGHT_EYE[0]], face_landmarks.landmark[RIGHT_EYE[3]])
            avg_ear = (left_ear + right_ear) / 2
            ear_history.append(avg_ear)

            # --- MAR & MOE (Yawns) ---
            mar = distance(face_landmarks.landmark[MOUTH[0]], face_landmarks.landmark[MOUTH[1]]) / distance(face_landmarks.landmark[MOUTH[2]], face_landmarks.landmark[MOUTH[3]])
            moe = mar / avg_ear if avg_ear > 0.01 else 10.0
            
            yawn_threshold = 0.6 
            if mar > yawn_threshold:
                yawn_history.append(1)
            else:
                yawn_history.append(0)
            
            yawn_freq = sum(yawn_history)

            # --- ADVANCED ML PREDICTION ---
            if model is not None:
                input_data = pd.DataFrame([[avg_ear, left_ear, right_ear, mar, moe, tilt_angle]], columns=["EAR", "Left_EAR", "Right_EAR", "MAR", "MOE", "Tilt"])
                
                if scaler is not None:
                    try:
                        input_data = scaler.transform(input_data)
                    except:
                        pass

                raw_ml_pred = model.predict(input_data)[0]
                
                if str(raw_ml_pred) == '0': raw_ml_pred = "Normal"
                elif str(raw_ml_pred) == '1': raw_ml_pred = "Sleepy"

                ml_history.append(raw_ml_pred)
                try: ml_pred = statistics.mode(ml_history)
                except: ml_pred = raw_ml_pred

            # --- ALGORITHMS ---
            closed_frames = sum(1 for e in ear_history if e < EYE_THRESHOLD)
            perclos = (closed_frames / len(ear_history)) * 100 if len(ear_history) > 0 else 0

            distracted_frames = sum(1 for h in head_history if "DISTRACTED" in h)
            distracted_percent = (distracted_frames / len(head_history)) * 100 if len(head_history) > 0 else 0

            fatigue_score = (perclos * 1.5) + (yawn_freq * 0.5) + (distracted_percent * 0.3)
            fatigue_score = round(min(fatigue_score, 100.0), 1)

            eyebrow_dist = distance(face_landmarks.landmark[55], face_landmarks.landmark[285])
            face_width_dist = distance(face_landmarks.landmark[LEFT_FACE], face_landmarks.landmark[RIGHT_FACE])
            frown_ratio = eyebrow_dist / face_width_dist if face_width_dist > 0 else 1.0
            stress_history.append(1 if frown_ratio < 0.16 else 0)
            is_stressed = (sum(stress_history) / len(stress_history) > 0.8) if len(stress_history) > 0 else False

            # --- ALERTS ---
            alert_active = False
            msg = ""
            color = (0, 255, 0)

            if is_stressed and not alert_active:
                msg = "HIGH STRESS / ANGRY DISTRACTION!"
                color = (0, 0, 255)
                alert_active = True
                if time.time() - last_log_time > 5:
                    log_to_database("Quarreling / Stress", "HIGH", perclos)
                    last_log_time = time.time()

            elif perclos > 40.0: 
                msg = "CRITICAL FATIGUE (PERCLOS >40%)"
                color = (0, 0, 255)
                alert_active = True
                if time.time() - last_log_time > 5:
                    log_to_database("Critical Sleep", "HIGH", perclos)
                    last_log_time = time.time()

            elif ml_pred == "Sleepy" and avg_ear < EYE_THRESHOLD:
                if EYE_CLOSE_START is None: EYE_CLOSE_START = time.time()
                
                if time.time() - EYE_CLOSE_START > DROWSY_TIME:
                    msg = "MICRO-SLEEP DETECTED!"
                    color = (0, 165, 255)
                    alert_active = True
                    if time.time() - last_log_time > 5:
                        log_to_database("Micro-Sleep", "MEDIUM", perclos)
                        last_log_time = time.time()
            else:
                EYE_CLOSE_START = None

            if distracted_percent > 70.0 and not alert_active:
                msg = f"ALERT! {min([h for h in head_history if 'DISTRACTED' in h], key=len)}"
                color = (255, 0, 0)
                alert_active = True
                if time.time() - last_log_time > 5:
                    log_to_database("Distraction_or_Phone", "WARNING", perclos)
                    last_log_time = time.time()

            if alert_active:
                cv2.putText(frame, msg, (50, 400), FONT_STYLE, ALERT_SCALE, color, 2)
                winsound.Beep(2500, 200)

            # --- GUI DASHBOARD IN CV2 ---
            cv2.rectangle(frame, (10, 10), (320, 200), (0, 0, 0), -1)
            cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (20, 40), FONT_STYLE, DASH_SCALE, (255, 255, 255), 1)
            cv2.putText(frame, f"XGBoost AI: {ml_pred}", (20, 70), FONT_STYLE, DASH_SCALE, (0, 255, 0) if ml_pred=="Normal" else (0,0,255), 1)
            cv2.putText(frame, f"Focus: {head_dir}", (20, 100), FONT_STYLE, DASH_SCALE, (255, 255, 255), 1)
            cv2.putText(frame, f"Stress: {'HIGH' if is_stressed else 'Normal'}", (20, 130), FONT_STYLE, DASH_SCALE, (0,0,255) if is_stressed else (255,255,255), 1)
            cv2.putText(frame, f"Fatigue Score: {fatigue_score}/100", (20, 160), FONT_STYLE, DASH_SCALE, (0,165,255), 1)
            cv2.putText(frame, f"Mode: {'NIGHT (Enhance)' if night_mode else 'NORMAL'}", (20, 190), FONT_STYLE, DASH_SCALE, (200,200,200), 1)

    cv2.imshow("Advanced Drowsiness Detection System", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('n'): night_mode = not night_mode

conn.close()
cap.release()
cv2.destroyAllWindows()
