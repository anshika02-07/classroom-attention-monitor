# Smart Classroom Attention Monitor

Real-time computer vision system that monitors student 
attention levels using a webcam.

## What it does
- Detects faces in real time using Haar Cascade classifier
- Analyses eye state of each detected face
- Attentive = 2 or more eyes detected (green box)
- Drowsy = fewer than 2 eyes detected (red box)  
- Shows live attention percentage on dashboard overlay
- Logs session data to CSV every second

## Sample Output
!(sample_output.png)

The CSV log captures timestamp, total faces, 
attentive count and attention percentage every second.

## Tech Stack
- Python 3
- OpenCV
- Haar Cascade Classifiers
- csv, datetime (Python standard library)

## How to run
pip install opencv-python numpy
python attention_monitor.py

Press Q to quit. 
Session data saved to attention_log.csv automatically.

## Limitations
- Performance drops in low light
- Side-profile faces may not be detected
- Haar cascades less robust than deep learning detectors
  — CNN-based detection is the planned next step
