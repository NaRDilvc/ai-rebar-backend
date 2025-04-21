from ultralytics import YOLO
import os

# Debug log
print("🧠 Loading YOLO model from model/best.pt...")
model_path = "model/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at: {model_path}")

model = YOLO(model_path)
print("✅ Model loaded successfully.")

def run_inference(image_path):
    print(f"📸 Running inference on: {image_path}")

    try:
        results = model(image_path)
        output = []

        for r in results:
            if hasattr(r, "probs") and r.probs is not None:
                class_id = r.probs.top1
                confidence = r.probs.top1conf
                output.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence)
                })
                print(f"✅ Prediction: Class {class_id}, Confidence {confidence:.2f}")
            else:
                print("⚠️ No classification probabilities found.")

        return {"results": output}

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return {"results": [], "error": str(e)}
