import cv2
import csv
import time
from datetime import datetime

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# ─── Setup CSV log file ────
log_file = open('attention_log.csv', 'w', newline='')
writer = csv.writer(log_file)
writer.writerow(['Timestamp', 'Total_Faces', 
                  'Attentive', 'Attention_%'])

# ─── Setup webcam ──
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ─── Variables ────
last_log_time = time.time()
prev_time     = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── FPS calculation ──
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ── Convert to grayscale ──
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect faces ──
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    total_faces  = len(faces)
    attentive    = 0

    # ── Process each face ───
    for (x, y, w, h) in faces:

        roi_gray  = gray[y:y+h,   x:x+w]
        roi_color = frame[y:y+h,  x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        eyes_open = len(eyes) >= 2

        if eyes_open:
            attentive += 1
            box_color  = (0, 255, 0)   # green = attentive
            label      = "Attentive"
        else:
            box_color  = (0, 0, 255)   # red = drowsy
            label      = "Drowsy"

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        # Draw label background bar
        cv2.rectangle(frame, (x, y-30), (x+w, y), box_color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (255, 255, 255), 1)

        # Draw eyes detected
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,
                         (ex, ey), (ex+ew, ey+eh),
                         (255, 255, 0), 1)

    # ── Calculate attention % ──
    attention_pct = 0
    if total_faces > 0:
        attention_pct = int((attentive / total_faces) * 100)

    # ── Dashboard overlay ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Dashboard text
    cv2.putText(frame, f"Faces     : {total_faces}",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Attentive : {attentive}",
                (10, 60), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (0, 255, 0), 1)

    # Attention % with colour
    pct_color = (0,255,0) if attention_pct >= 50 else (0,0,255)
    cv2.putText(frame, f"Attention : {attention_pct}%",
                (10, 90), cv2.FONT_HERSHEY_DUPLEX,
                0.7, pct_color, 1)
    cv2.putText(frame, f"FPS       : {fps:.1f}",
                (10, 120), cv2.FONT_HERSHEY_DUPLEX,
                0.6, (180, 180, 180), 1)

    # ── Low attention alert ───
    if attention_pct < 50 and total_faces > 0:
        cv2.putText(frame, "! LOW ATTENTION !",
                    (350, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (0, 0, 255), 2)

    # ── Log to CSV every 1 second
    if time.time() - last_log_time >= 1:
        timestamp = datetime.now().strftime("%H:%M:%S")
        writer.writerow([timestamp, total_faces,
                          attentive, attention_pct])
        last_log_time = time.time()

    
    cv2.imshow("Attention Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


log_file.close()
cap.release()
cv2.destroyAllWindows()
print("Session saved to attention_log.csv")