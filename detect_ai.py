from picamera2 import Picamera2
import time
import numpy as np
import cv2

# Kamera ve AI model yapılandırması
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
config["aiq"] = {
    "tensor_mode": "balloon"  # Klasör adı
}
picam2.configure(config)
picam2.start()
time.sleep(1)

print("AI model başlatıldı. Balonlar tespit ediliyor...")

while True:
    frame = picam2.capture_array()
    metadata = picam2.capture_metadata()

    # AI Tensor çıktısı alınır
    tensors = metadata.get("Tensor")
    if tensors:
        for det in tensors:
            # Tensor çıktısı: [cx, cy, w, h, obj_score, class_score, class_id]
            cx, cy, w, h, obj_score, cls_score, cls_id = det
            if obj_score * cls_score > 0.5:
                x1 = int((cx - w / 2) * frame.shape[1])
                y1 = int((cy - h / 2) * frame.shape[0])
                x2 = int((cx + w / 2) * frame.shape[1])
                y2 = int((cy + h / 2) * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Balloon", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Balloon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
