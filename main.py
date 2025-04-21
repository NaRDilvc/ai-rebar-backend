from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from yolo_infer import run_inference
import shutil
import os

app = FastAPI()

# Enable CORS (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_inference(temp_path)
    os.remove(temp_path)
    return result
