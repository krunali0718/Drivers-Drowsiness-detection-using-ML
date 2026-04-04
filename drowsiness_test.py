import cv2
import mediapipe as mp
import time
import math
import winsound
import joblib
import pandas as pd

# Load ML Model
model = joblib.load("drowsiness_model.pkl")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [13,14,78,308]

NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454
CHIN = 152

def distance(p1,p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

# ---------------- VARIABLES ---------------- #

# Replacing unused blink/yawn counters with MAR feature

tilt_angle = 0

tilt_angle = 0

# 🔥 TIME TRACKERS
EYE_CLOSE_START = None
HEAD_TILT_START = None

EYE_CLOSE_TIME = 0
HEAD_TILT_TIME = 0

EYE_THRESHOLD = 0.18        # LOWERED (important)
DROWSY_TIME = 2             # seconds
HEAD_TIME = 2               # seconds

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ---------- HEAD ---------- #

            nose = face_landmarks.landmark[NOSE]
            left_face = face_landmarks.landmark[LEFT_FACE]
            right_face = face_landmarks.landmark[RIGHT_FACE]
            chin = face_landmarks.landmark[CHIN]

            nose_x = int(nose.x * w)
            left_x = int(left_face.x * w)
            right_x = int(right_face.x * w)
            chin_y = int(chin.y * h)

            face_center = (left_x + right_x) // 2

            if nose_x < face_center - 40:
                head_direction = "LEFT"
                tilt_angle = -20

            elif nose_x > face_center + 40:
                head_direction = "RIGHT"
                tilt_angle = 20

            elif int(nose.y*h) > chin_y - 25:
                head_direction = "DOWN"
                tilt_angle = 25

            else:
                head_direction = "CENTER"
                tilt_angle = 0

            # ---------- EYES ---------- #

            left_eye_ratio = distance(
                face_landmarks.landmark[LEFT_EYE[1]],
                face_landmarks.landmark[LEFT_EYE[5]]
            ) / distance(
                face_landmarks.landmark[LEFT_EYE[0]],
                face_landmarks.landmark[LEFT_EYE[3]]
            )

            right_eye_ratio = distance(
                face_landmarks.landmark[RIGHT_EYE[1]],
                face_landmarks.landmark[RIGHT_EYE[5]]
            ) / distance(
                face_landmarks.landmark[RIGHT_EYE[0]],
                face_landmarks.landmark[RIGHT_EYE[3]]
            )

            eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # ---------- MOUTH (MAR) ---------- #

            mouth_ratio = distance(
                face_landmarks.landmark[MOUTH[0]],
                face_landmarks.landmark[MOUTH[1]]
            ) / distance(
                face_landmarks.landmark[MOUTH[2]],
                face_landmarks.landmark[MOUTH[3]]
            )

            # ---------- EYE TIME ---------- #

            if eye_ratio < EYE_THRESHOLD:
                if EYE_CLOSE_START is None:
                    EYE_CLOSE_START = time.time()

                EYE_CLOSE_TIME = time.time() - EYE_CLOSE_START
            else:
                EYE_CLOSE_START = None
                EYE_CLOSE_TIME = 0

            # ---------- HEAD TIME ---------- #

            if head_direction != "CENTER":
                if HEAD_TILT_START is None:
                    HEAD_TILT_START = time.time()

                HEAD_TILT_TIME = time.time() - HEAD_TILT_START
            else:
                HEAD_TILT_START = None
                HEAD_TILT_TIME = 0

            # ---------- ML ---------- #

            moe = mouth_ratio / eye_ratio if eye_ratio > 0.01 else 10.0

            input_data = pd.DataFrame(
                [[eye_ratio, left_eye_ratio, right_eye_ratio, mouth_ratio, moe, tilt_angle]],
                columns=["EAR", "Left_EAR", "Right_EAR", "MAR", "MOE", "Tilt"]
            )

            prediction = model.predict(input_data)[0]

            # ---------- FINAL ALERT ---------- #

            if (EYE_CLOSE_TIME > DROWSY_TIME) or (HEAD_TILT_TIME > HEAD_TIME):

                cv2.putText(frame, "DROWSINESS ALERT!", (150,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,0,255), 3)

                winsound.Beep(2500, 500)

            # ---------- DISPLAY ---------- #

            cv2.putText(frame,f"EAR: {round(eye_ratio,2)}",(30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            cv2.putText(frame,f"Head: {head_direction}",(30,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            cv2.putText(frame,f"ML: {prediction}",(30,120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()