import time
import cv2
from picamera2 import Picamera2

# Sınıf etiketleri
class_map = {
    0: "red_balloon",
    1: "blue_balloon"
}

# Kamera başlat
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (224, 224)})
picam2.configure(config)
picam2.start()

frame_count = 0
start_time = time.time()

print("Kamera çalışıyor. Çıkmak için q tuşuna bas.")

while True:
    frame = picam2.capture_array()
    
    # AI kamera metadata (inference sonucu)
    metadata = picam2.capture_metadata()
    inference = metadata.get("Inference", None)

    label_text = "No detection"
    if inference:
        result = inference[0]
        class_id = int(result.get("class", -1))
        confidence = result.get("confidence", 0.0)
        if class_id in class_map:
            label_text = f"{class_map[class_id]} ({confidence:.2f})"

    # Ekrana yaz
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # FPS hesapla
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        fps = frame_count / elapsed
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
        frame_count = 0
        start_time = time.time()

    cv2.imshow("AI Balloon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
