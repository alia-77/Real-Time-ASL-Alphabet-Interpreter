# Real-Time-ASL-Alphabet-Interpreter

A real-time American Sign Language fingerspelling interpreter. Uses MediaPipe to extract hand landmarks from a webcam feed, classifies the hand shape with a Random Forest model, and stabilizes predictions over time to build up text letter by letter.

Scope: this recognizes static ASL alphabet signs (A-Z, plus a "no sign" blank class), not continuous or sentence-level sign language. That's a different problem, since it requires modeling motion over time rather than classifying a single frame. See Future Work below.

## How it works

**1. Feature extraction.** MediaPipe's HandLandmarker detects 21 hand landmarks per frame. Each landmark's (x, y) is made relative to the wrist (landmark 0), so the model learns hand shape rather than where the hand happens to be in the frame.

**2. Classification.** A Random Forest trained on the resulting 42-dimensional feature vector predicts a letter per frame.

**3. Stabilization.** Raw per-frame predictions are noisy, so the app keeps a rolling window of the last 10 predictions and takes the majority vote as the "stable" letter. That letter has to hold for about 45 consecutive frames (roughly 1.5 seconds at 30fps) before it locks into the output text. This stops a single flickering misclassification from typing the wrong letter.

**4. Live UI.** OpenCV overlay showing the current predicted sign, a progress bar for how long it's been held, and the sentence built so far. Keyboard shortcuts let you clear (`c`) or add a space (`s`).

## Results

Trained on 13,928 labeled samples across 27 classes (A-Z plus blank). Random Forest with 100 trees hit 98.1% accuracy on a held-out 20% test split.

## Project structure
```
ASL.ipynb              # training pipeline: landmark extraction from labeled images, then model training
ASL_Detection.py       # real-time webcam app, loads the trained model and runs live inference
asl_landmarks.csv      # extracted landmark features and labels used to train the model
asl_model.pkl          # trained Random Forest model
hand_landmarker.task   # MediaPipe's pretrained hand landmark detection model
```

## Run it
```bash
pip install -r requirements.txt
python ASL_Detection.py
```
Keys: `q` to quit, `c` to clear the sentence, `s` to add a space.

## Future work
Extending to continuous sign language recognition (full ASL, not just fingerspelling) would need temporal modeling instead of single-frame classification. Two-handed sign support is also not implemented yet.

## Stack
Python, OpenCV, MediaPipe, scikit-learn (Random Forest)
