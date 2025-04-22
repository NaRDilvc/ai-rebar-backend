from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import requests

app = FastAPI()

# Enable CORS (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔗 Your external inference server (to be replaced later)
INFERENCE_URL = "http://localhost:9000/infer"  # Replace with your container endpoint

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔁 Send image to external inference container
    with open(temp_path, "rb") as img_file:
        files = {"file": (file.filename, img_file, file.content_type)}
        try:
            response = requests.post(INFERENCE_URL, files=files)
            result = response.json()
        except Exception as e:
            result = {"error": f"Failed to contact inference server: {e}"}

    os.remove(temp_path)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
