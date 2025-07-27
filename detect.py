import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# MODEL ve LABEL dosyası
MODEL_PATH = "balloon_model_imx500.tflite"
LABEL = "Balloon"

# TFLite modeli yükle
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Kamera başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Giriş görüntüsünü modele uygun hale getir
    input_image = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_image, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Çıktı verilerini al
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * frame.shape[1]), int(ymin * frame.shape[0])
            x2, y2 = int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, LABEL, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Balloon Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
