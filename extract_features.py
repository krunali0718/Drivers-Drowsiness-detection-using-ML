import cv2
import mediapipe as mp
import pandas as pd
import math
import glob
import os
import random

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308] # 13: upper lip, 14: lower lip, 78: left corner, 308: right corner
NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454
CHIN = 152

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_features(image_path, label):
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    h, w, _ = image.shape
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # ---------- HEAD TILT ---------- #
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
                tilt_angle = -20 # Left
            elif nose_x > face_center + 40:
                tilt_angle = 20  # Right
            elif int(nose.y * h) > chin_y - 25:
                tilt_angle = 25  # Down
            else:
                tilt_angle = 0   # Center

            # ---------- EYE ASPECT RATIO (EAR) ---------- #
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

            # ---------- MOUTH ASPECT RATIO (MAR) ---------- #
            mouth_ratio = distance(
                face_landmarks.landmark[MOUTH[0]], # Upper lip
                face_landmarks.landmark[MOUTH[1]]  # Lower lip
            ) / distance(
                face_landmarks.landmark[MOUTH[2]], # Left corner
                face_landmarks.landmark[MOUTH[3]]  # Right corner
            )

            # ---------- MOE (Mouth Over Eye) ---------- #
            moe = mouth_ratio / eye_ratio if eye_ratio > 0.01 else 10.0

            return [eye_ratio, left_eye_ratio, right_eye_ratio, mouth_ratio, moe, tilt_angle, label]
            
    return None

def process_dataset():
    data = []
    MAX_IMAGES = 4000

    # Process Drowsy images
    print(f"Processing up to {MAX_IMAGES} Drowsy images (randomized) ...")
    drowsy_images = glob.glob("dataset/Drowsy/*.png") + glob.glob("dataset/Drowsy/*.jpg")
    random.shuffle(drowsy_images)
    for i, img in enumerate(drowsy_images):
        if i >= MAX_IMAGES: break
        if i % 100 == 0: print(f"Drowsy: {i}/{MAX_IMAGES}")
        features = extract_features(img, "Sleepy")
        if features:
            data.append(features)

    # Process Non Drowsy images
    print(f"\nProcessing up to {MAX_IMAGES} Non Drowsy images (randomized)...")
    non_drowsy_images = glob.glob("dataset/Non Drowsy/*.png") + glob.glob("dataset/Non Drowsy/*.jpg")
    random.shuffle(non_drowsy_images)
    for i, img in enumerate(non_drowsy_images):
        if i >= MAX_IMAGES: break
        if i % 100 == 0: print(f"Non Drowsy: {i}/{MAX_IMAGES}")
        features = extract_features(img, "Normal")
        if features:
            data.append(features)

    df = pd.DataFrame(data, columns=["EAR", "Left_EAR", "Right_EAR", "MAR", "MOE", "Tilt", "Label"])
    df.to_csv("dataset_features.csv", index=False)
    print(f"\nExtraction complete! Saved {len(df)} records to dataset_features.csv")

if __name__ == "__main__":
    process_dataset()
