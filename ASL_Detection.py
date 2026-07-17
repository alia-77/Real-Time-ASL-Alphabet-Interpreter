import cv2
import mediapipe as mp
import pickle
import numpy as np
import warnings
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 0. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)

# 1. LOAD THE BRAIN
with open('asl_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. SETUP CAMERA & MEDIAPIPE
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# VARIABLES FOR SENTENCE BUILDING
prediction_history = []
current_sentence = ""
last_stable_letter = ""
counter = 0  # To track how long a sign is held

cap = cv2.VideoCapture(0)
print("Translator LIVE! Keys: 'q'=Quit, 'c'=Clear, 's'=Space")

SKELETON_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        coords = []
        hand = result.hand_landmarks[0]
        base_x, base_y = hand[0].x, hand[0].y

        # DRAW SKELETON (White)
        for connection in SKELETON_CONNECTIONS:
            pt1 = (int(hand[connection[0]].x * w), int(hand[connection[0]].y * h))
            pt2 = (int(hand[connection[1]].x * w), int(hand[connection[1]].y * h))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

        for landmark in hand:
            coords.append(landmark.x - base_x)
            coords.append(landmark.y - base_y)
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 5, (0, 255, 0), -1)
        
        # PREDICT & STABILIZE
        input_data = np.array([coords])
        raw_prediction = model.predict(input_data)[0]
        prediction_history.append(raw_prediction)
        if len(prediction_history) > 10: prediction_history.pop(0)
        
        stable_letter = max(set(prediction_history), key=prediction_history.count)

        # SENTENCE LOGIC: If held for 45 frames (~1.5s), add to sentence
        if stable_letter == last_stable_letter:
            counter += 1
        else:
            counter = 0
            last_stable_letter = stable_letter

        if counter == 45: # "Lock in" the letter
            current_sentence += stable_letter
            counter = 0 # Reset to prevent repeating instantly

        # --- UI DESIGN: TOP OVERLAY ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1) # Dark Top Bar
        cv2.rectangle(overlay, (0, h-80), (w, h), (20, 20, 20), -1) # Dark Bottom Bar
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Top Text (Live Prediction)
        cv2.putText(frame, f"SIGN: {stable_letter}", (20, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        
        # Progress Bar (visualizes the hold time)
        bar_width = int((counter / 45) * 200)
        cv2.rectangle(frame, (w-250, 35), (w-250 + bar_width, 45), (0, 255, 0), -1)
        cv2.rectangle(frame, (w-250, 35), (w-50, 45), (255, 255, 255), 1)

    # --- UI DESIGN: BOTTOM OVERLAY (Sentence) ---
    cv2.putText(frame, f"TEXT: {current_sentence}", (20, h-30), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)

    cv2.imshow('ASL Real-Time Translator', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): current_sentence = "" # Clear
    if key == ord('s'): current_sentence += " " # Space

cap.release()
cv2.destroyAllWindows()