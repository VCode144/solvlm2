import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText

from models import SessionLocal, FrameMetadata
import sys

# === Restrictions ===
default_restricted_objects = {"knife", "gun", "dog", "motorbike", "truck"}
default_restricted_actions = {
    "pointing", "holding a knife", "threatening", "stabbing", "firing", "shooting",
    "without gloves", "without hairnet", "without mask", "topping pizza without gloves"
}
hygiene_keywords = {"no gloves", "no mask", "no hairnet", "bare hands", "unsanitary", "dirty", "not wearing protective gear"}

def load_models():
    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path)
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    yolo = YOLO("yolov8n.pt")
    return processor, vlm, yolo, device

def process_video(
    video_path: str,
    video_id: str,
    restricted_objects: set = None,
    restricted_actions: set = None,
    processor=None,
    vlm=None,
    yolo=None,
    device=None,
    question: str = None
):
    restricted_objects = restricted_objects or default_restricted_objects
    restricted_actions = restricted_actions or default_restricted_actions
    if processor is None or vlm is None or yolo is None or device is None:
        processor, vlm, yolo, device = load_models()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"[INFO] Video duration: {duration:.2f}s at {fps} FPS")
    session = SessionLocal()
    print(session.query(FrameMetadata).count())
    descriptions = []
    alerts = []
    current_frame = 0
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()
        if not success:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        # === YOLO Object Detection ===
        results = yolo.predict(frame, verbose=False)[0]
        detected_labels = [results.names[int(cls)] for cls in results.boxes.cls]
        yolo_matches = [label for label in detected_labels if label in restricted_objects]
        # === VLM Description ===
        prompt = question or (
            "What is happening in this kitchen frame? Describe chef's hygiene: "
            "Are they wearing gloves, mask, hairnet? Is anyone doing food prep without proper hygiene?"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        with torch.no_grad():
            output_ids = vlm.generate(**inputs, do_sample=False, max_new_tokens=128)
        description = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        timestamp = round(current_frame / fps, 2)
        # === Alert check ===
        vlm_objects = [obj for obj in restricted_objects if obj.lower() in description.lower()]
        vlm_actions = [act for act in restricted_actions if act.lower() in description.lower()]
        hygiene_flags = [kw for kw in hygiene_keywords if kw.lower() in description.lower()]
        all_violations = list(set(yolo_matches + vlm_objects + vlm_actions + hygiene_flags))
        alert = None
        if all_violations:
            alert = f"🚨 ALERT: Hygiene/Restricted content {all_violations} at {timestamp}s"
            alerts.append(f"{timestamp}s - {alert}")
            print(alert)
        # === Save to DB ===
        frame_meta = FrameMetadata(
            video_id=video_id,
            timestamp=timestamp,
            frame_number=frame_idx,
            description=description,
            alert=alert if 'alert' in locals() else None
        )
        session.add(frame_meta)
        session.commit()
        print(f"[DB] Frame saved: {timestamp}s - {description[:60]}...")
        print(f"[LIVE] {timestamp}s: {description}")
        print(f"[{timestamp}s] {description}")
        descriptions.append({
            "timestamp": timestamp,
            "frame": frame_idx,
            "description": description,
            "alert": alert
        })
        current_frame += fps
        frame_idx += 1
    cap.release()
    session.close()
    return {
        "video_path": video_path,
        "duration": duration,
        "fps": fps,
        "frames_analyzed": frame_idx,
        "descriptions": descriptions,
        "alerts": alerts
    }

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("❌ Error: Expected 4 arguments: video_path, restricted_objects, question, video_id")
        sys.exit(1)
    video_path = sys.argv[1]
    restricted_objects = sys.argv[2].split(",")
    question = sys.argv[3]
    video_id = sys.argv[4]
    process_video(
        video_path=video_path,
        video_id=video_id,
        restricted_objects=set(restricted_objects),
        question=question
        # restricted_actions will use the default
    )
