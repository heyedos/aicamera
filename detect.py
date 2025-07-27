from picamera2 import Picamera2
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

MODEL_PATH = "balloon_model_imx500.tflite"
LABEL = "Balloon"
THRESHOLD = 0.5  # Tespit eşiği

# Modeli yükle
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Kamera başlat
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (width, height)})
picam2.configure(config)
picam2.start()
time.sleep(1)

prev_time = time.time()

while True:
    frame = picam2.capture_array()
    input_image = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_image, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    detections = interpreter.get_tensor(output_details[0]['index'])[0]  # [6300, 7]

    balloon_detected = False

    for det in detections:
        cx, cy, w, h, obj_score, class_score, class_id = det
        score = obj_score * class_score

        if score > THRESHOLD:
            x1 = int((cx - w / 2) * frame.shape[1])
            y1 = int((cy - h / 2) * frame.shape[0])
            x2 = int((cx + w / 2) * frame.shape[1])
            y2 = int((cy + h / 2) * frame.shape[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, LABEL, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            balloon_detected = True
            print(f"🎈 Balloon Detected at ({x1}, {y1}, {x2}, {y2}) with score {score:.2f}")

    # FPS hesapla
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if balloon_detected:
        cv2.putText(frame, "Balloon Detected!", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Balloon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
