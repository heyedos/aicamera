# imx500.py

from libcamera import controls
import time

class CameraInference:
    def __init__(self, picam2, model_path):
        # Kamera zaten başlatılmış olarak alınır (çakışmayı önler)
        self.picam2 = picam2
        self.model_path = model_path

        # Kamera ayarlarını inference moduna geçir
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "AfTrigger": 0,
            "InferenceConfig": self.model_path,
            "InferenceMode": 2,
        })

        time.sleep(1)  # Kamera AI işlemciyi başlatsın

    def run_inference(self):
        while True:
            metadata = self.picam2.capture_metadata()
            if "InferenceResult" in metadata:
                yield metadata["InferenceResult"]
            else:
                yield {}
            time.sleep(0.04)  # yaklaşık 25 FPS
