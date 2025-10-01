from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from pathlib import Path
import subprocess
import shutil
import uuid
import os
import cv2

def convert_avi_to_mp4(avi_path, mp4_path):
    command = [
        "ffmpeg",
        "-y",                      # overwrite if exists
        "-i", avi_path,            # input file
        "-c:v", "libx264",         # H.264 codec
        "-preset", "fast",
        "-pix_fmt", "yuv420p",     # browser compatibility (IMPORTANT!)
        "-crf", "23",              # quality control
        mp4_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode == 0:
        print(f"✅ Converted {avi_path} → {mp4_path}")
    else:
        print(f"❌ FFmpeg failed: {result.stderr.decode()}")

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI()

# Serve static files (CSS, JS, images, uploads)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (HTML)
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Load YOLO model at startup
# ---------------------------
MODEL_PATH = "models/best.pt"
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = None
    print("⚠️ Model not found! Place best.pt inside models/")

# ---------------------------
# Landing Page
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------
# Image Upload Page
# ---------------------------
@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request):
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/image", response_class=HTMLResponse)
async def image_upload(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    upload_path = f"static/uploads/{unique_filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(source=upload_path, conf=0.5, save=True, project="static/uploads", name="preds", exist_ok=True)

    # YOLO saves prediction with same filename inside static/uploads/preds
    base_name = Path(upload_path).stem  
    pred_img_url = f"/static/uploads/preds/{base_name}.jpg"

    return templates.TemplateResponse("image.html", {
        "request": request,
        "uploaded_file": f"/{upload_path}",   # original upload
        "pred_file": pred_img_url             # predicted image (web-safe path)
    })

# ---------------------------
# Video Upload Page
# ---------------------------
@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})

@app.post("/video", response_class=HTMLResponse)
async def video_upload(request: Request, file: UploadFile = File(...)):
    # Save uploaded video
    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    upload_path = f"static/uploads/{unique_filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO on video
    results = model.predict(
        source=upload_path,
        conf=0.5,
        save=True,
        project="static/uploads",
        name="preds",
        exist_ok=True
    )

    # YOLO saves prediction with same name but always as .avi
    base_name = Path(upload_path).stem
    avi_path = f"static/uploads/preds/{base_name}.avi"
    mp4_path = f"static/uploads/preds/{base_name}.mp4"

    # convert avi → mp4
    convert_avi_to_mp4(avi_path, mp4_path)

    pred_video_url = f"/static/uploads/preds/{base_name}.mp4"   # serve the mp4 to frontend

    return templates.TemplateResponse("video.html", {
        "request": request,
        "uploaded_video": f"/{upload_path}",
        "pred_video": pred_video_url
    })

