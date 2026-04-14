import math

# Facial Landmarks Indices based on MediaPipe FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308] # 13: upper lip, 14: lower lip, 78: left corner, 308: right corner
NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454
CHIN = 152

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_tilt_angle(face_landmarks, w, h):
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
        return -20 # Looking Left
    elif nose_x > face_center + 40:
        return 20  # Looking Right
    elif int(nose.y * h) > chin_y - 25:
        return 25  # Looking Down
    return 0       # Looking Center

def calculate_ear(face_landmarks):
    # Left Eye
    left_eye_ratio = distance(
        face_landmarks.landmark[LEFT_EYE[1]], face_landmarks.landmark[LEFT_EYE[5]]
    ) / distance(
        face_landmarks.landmark[LEFT_EYE[0]], face_landmarks.landmark[LEFT_EYE[3]]
    )

    # Right Eye
    right_eye_ratio = distance(
        face_landmarks.landmark[RIGHT_EYE[1]], face_landmarks.landmark[RIGHT_EYE[5]]
    ) / distance(
        face_landmarks.landmark[RIGHT_EYE[0]], face_landmarks.landmark[RIGHT_EYE[3]]
    )

    return (left_eye_ratio + right_eye_ratio) / 2

def calculate_mar(face_landmarks):
    return distance(
        face_landmarks.landmark[MOUTH[0]], face_landmarks.landmark[MOUTH[1]]
    ) / distance(
        face_landmarks.landmark[MOUTH[2]], face_landmarks.landmark[MOUTH[3]]
    )

def extract_features(face_landmarks, w, h):
    """
    Extracts geometric features from the 3D FaceMesh configuration.
    """
    ear = calculate_ear(face_landmarks)
    mar = calculate_mar(face_landmarks)
    tilt = calculate_tilt_angle(face_landmarks, w, h)
    
    # MOE (Mouth Over Eye) ratio
    moe = mar / ear if ear > 0.01 else 10.0
    
    return {
        "EAR": ear,
        "MAR": mar,
        "MOE": moe,
        "Tilt": tilt
    }
