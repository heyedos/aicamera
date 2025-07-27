from picamera2 import Picamera2
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

MODEL_PATH = "balloon_model_imx500.tflite"
LABEL = "Balloon"

# TFLite modeli yükle
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Giriş / çıkış tensor bilgileri
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug çıktısı
print("Model giriş şekli:", input_details[0]['shape'])
print("Model çıkışları:")
for i, detail in enumerate(output_details):
    print(f"{i}: name={detail['name']} shape={detail['shape']}")

# Giriş boyutları
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Kamera başlat
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (width, height)})
picam2.configure(config)
picam2.start()
time.sleep(1)

try:
    while True:
        frame = picam2.capture_array()

        # Model girişine uygun hale getir
        input_image = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(input_image, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Çıkış tensor verilerini al
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
        except IndexError:
            print("Çıkış tensör sayısı 3 değil! Lütfen modelin çıkış yapısını kontrol edin.")
            break

        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1 = int(xmin * frame.shape[1]), int(ymin * frame.shape[0])
                x2, y2 = int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, LABEL, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Balloon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
