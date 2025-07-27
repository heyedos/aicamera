import time
import cv2
import numpy as np
from imx500 import Imx500Pipeline

# Kamera çözünürlüğü modeli eğitirkenki ile aynı olmalı (320x320)
pipeline = Imx500Pipeline({
    "model_type": "object_detection",
    "tensor_mode": "balloon",  # /lib/firmware/imx500/model/balloon klasörü
    "threads": 2
})

pipeline.start()
print("Kamera başlatıldı, AI modeli yükleniyor...")

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    img, result = pipeline.get_frame()
    frame_count += 1

    # FPS hesapla
    current_time = time.time()
    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0

    balloon_detected = False

    if result is not None and "objects" in result:
        for obj in result["objects"]:
            label = obj.get("label", "object")
            score = obj.get("score", 0)
            box = obj.get("bbox", [0, 0, 0, 0])  # [x1, y1, x2, y2]

            if score > 0.5 and label.lower() == "balloon":
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                balloon_detected = True

    if not balloon_detected:
        cv2.putText(img, "No balloon detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS göster
    cv2.putText(img, f"FPS: {fps:.1f}", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Balloon Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
