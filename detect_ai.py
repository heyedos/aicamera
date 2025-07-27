import cv2
import json
import time
import os
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("AI model başlatıldı. Balonlar tespit ediliyor...")

# FPS hesaplama
prev_time = time.time()
frame_count = 0
fps = 0

while True:
    frame = picam2.capture_array()
    frame_count += 1

    # FPS güncelle
    current_time = time.time()
    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        frame_count = 0
        prev_time = current_time

    # AI sonucu JSON dosyasından oku
    try:
        with open("/run/shm/ai_toolkit/output", "r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    objects = data.get("objects", [])
    balloon_detected = False

    for obj in objects:
        label = obj.get("label", "")
        score = obj.get("score", 0)
        x1, y1, x2, y2 = map(int, obj.get("bbox", [0, 0, 0, 0]))

        if label.lower() == "balloon" and score > 0.5:
            balloon_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not balloon_detected:
        cv2.putText(frame, "No balloon detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # FPS ekle
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Balloon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
