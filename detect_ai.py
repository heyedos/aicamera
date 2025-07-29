# detect_ai.py

import argparse
import time
import cv2
from picamera2 import Picamera2
from imx500 import CameraInference

# Komut satırı argümanları
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='network.rpk')
parser.add_argument('--labels', type=str, default='labels.txt')
args = parser.parse_args()

# Etiketleri oku
with open(args.labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Kamera başlat
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Inference sınıfını başlat (kamera objesini paylaşıyoruz!)
inference = CameraInference(picam2, args.model)

# FPS ölçüm başlangıcı
frame_count = 0
start_time = time.time()

# Ana döngü
for result in inference.run_inference():
    frame = picam2.capture_array()

    detections = result.get("detections", [])

    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        label = labels[det["class"]]
        score = det["score"]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
