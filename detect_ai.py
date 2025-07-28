import time
from imx500_sdk import InferenceEngine, CameraInput

# 📁 Yüklemek istediğin .rpk model dosyasının yolu
MODEL_PATH = "/home/pi/output_rpk/network.rpk"

# 🎬 Kamera başlat
camera = CameraInput(camera_id=0, resolution=(640, 480), framerate=30)
camera.start()

# 🧠 Model yükle
engine = InferenceEngine(model_path=MODEL_PATH)

print("✅ Model yüklendi. İnferans başlıyor...")

try:
    while True:
        # 📷 Kameradan bir kare al
        frame = camera.read()

        # 🔍 Modeli çalıştır
        result = engine.infer(frame)

        # 📊 Tahmin sonuçlarını yazdır
        print("🔎 Tahmin sonucu:", result)

        time.sleep(0.5)

except KeyboardInterrupt:
    print("🛑 İnferans durduruldu.")

finally:
    camera.stop()