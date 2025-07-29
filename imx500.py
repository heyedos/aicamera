# imx500.py

class CameraInference:
    def __init__(self, model_path):
        from libcamera import controls
        from picamera2 import Picamera2
        import time

        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        self.picam2.configure(self.config)
        self.picam2.start()

        time.sleep(2)  # kamera ısınsın

        # AI modeli yükle
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "AfTrigger": 0,
            "InferenceMode": 2,
            "InferenceConfig": model_path,
        })

    def run_inference(self):
        import time
        while True:
            metadata = self.picam2.capture_metadata()
            if "InferenceResult" in metadata:
                yield metadata["InferenceResult"]
            else:
                yield {}
            time.sleep(0.04)  # yaklaşık 25 fps
