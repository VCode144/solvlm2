
# Fast API code for web 



from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil, os, uuid
import subprocess
import uvicorn
import sys

from models import SessionLocal, FrameMetadata
from process_video import process_video, load_models

app = FastAPI()

# Allow CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Upload Video ===
@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    restricted_objects: str = Form(...),
    question: str = Form("What is happening in this kitchen frame? Describe chef's hygiene..."),
    background_tasks: BackgroundTasks = None
):
    video_id = str(uuid.uuid4())
    video_path = f"uploads/{video_id}.mp4"

    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[UPLOAD] Saved video to: {video_path}")
    print(f"[UPLOAD] Processing in background...")

    # Load models once and pass to background task
    processor, vlm, yolo, device = load_models()
    background_tasks.add_task(
        process_video,
        video_path,
        video_id,
        set(restricted_objects.split(",")),
        None,  # use default restricted_actions
        processor,
        vlm,
        yolo,
        device,
        question
    )

    return {"message": "Processing started", "video_id": video_id}


# === Query by Time ===
@app.get("/frames")
def get_descriptions(start: float, end: float):
    session = SessionLocal()
    results = session.query(FrameMetadata).filter(
        FrameMetadata.timestamp >= start,
        FrameMetadata.timestamp <= end
    ).all()
    session.close()
    return [{"time": f.timestamp, "desc": f.description, "alert": f.alert} for f in results]

# === Search by Keyword ===
@app.get("/search")
def search_description(q: str):
    session = SessionLocal()
    results = session.query(FrameMetadata).filter(FrameMetadata.description.ilike(f"%{q}%")).all()
    session.close()
    return [{"time": f.timestamp, "desc": f.description, "alert": f.alert} for f in results]



    
@app.get("/status")
def check_status(video_id: str):
    session = SessionLocal()
    frames = session.query(FrameMetadata).filter(FrameMetadata.video_id == video_id).order_by(FrameMetadata.timestamp).all()

    # All frame descriptions
    frame_data = [{"time": f.timestamp, "desc": f.description, "alert": f.alert} for f in frames]

    # Alerts
    alerts = [f.alert for f in frames if f.alert]

    session.close()
    return {"frames": frame_data, "alerts": alerts}



# === Serve frontend ===
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()
    







if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
