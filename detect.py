from picamera2 import Picamera2
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

MODEL_PATH = "balloon_model_imx500.tflite"
LABEL = "Balloon"
CONFIDENCE_THRESHOLD = 0.5

# TFLite modeli yükle
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Giriş boyutları
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Model yapısını yazdır
print(f"Model giriş şekli: {input_details[0]['shape']}")
print("Model çıkışları:")
for i, detail in enumerate(output_details):
    print(f"{i}: name={detail['name']} shape={detail['shape']}")

# Kamera başlat
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (width, height)})
picam2.configure(config)
picam2.start()
time.sleep(1)

try:
    while True:
        frame = picam2.capture_array()

        # Giriş resmi boyutlandır
        input_image = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(input_image, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Modelin tek çıkışı: [1, 6300, 7]
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (6300, 7)

        for det in output_data:
            cx, cy, w, h, obj_score, class_score = det
            confidence = obj_score * class_score

            if confidence > CONFIDENCE_THRESHOLD:
                # Normalize'dan piksele çevir
                x1 = int((cx - w / 2) * frame.shape[1])
                y1 = int((cy - h / 2) * frame.shape[0])
                x2 = int((cx + w / 2) * frame.shape[1])
                y2 = int((cy + h / 2) * frame.shape[0])

                # Kutucuk çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{LABEL} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Balloon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
