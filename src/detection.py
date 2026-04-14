import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Solutions
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
except Exception as e:
    print(f"Error initializing MediaPipe FaceMesh: {e}")
    face_mesh = None

def get_face_landmarks(frame_rgb):
    """
    Process the RGB frame through MediaPipe to extract FaceMesh landmarks.
    Returns:
        multi_face_landmarks object or None
    """
    if face_mesh is None:
        return None
    results = face_mesh.process(frame_rgb)
    return results.multi_face_landmarks

def crop_face_for_cnn(frame, face_landmarks, w, h):
    """
    Given the full frame and exact face landmarks, crop the bounding box.
    This creates an optimal input specifically sized for the Convolutional Neural Network.
    Expand the bounding box slightly using dynamic margins.
    """
    x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
    
    # Create an adaptive bounding box padding matching the CNN's dataset shape
    margin_x = int(0.12 * (max(x_coords) - min(x_coords)))
    margin_y = int(0.12 * (max(y_coords) - min(y_coords)))
    
    x_min = max(0, min(x_coords) - margin_x)
    x_max = min(w, max(x_coords) + margin_x)
    y_min = max(0, min(y_coords) - int(margin_y * 1.5)) # Expand Top
    y_max = min(h, max(y_coords) + margin_y)
    
    # Must have a valid physical size
    if (x_max - x_min) > 20 and (y_max - y_min) > 20: 
        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size != 0:
            return face_crop, (x_min, y_min, x_max, y_max)
            
    return None, None
