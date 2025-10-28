from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
from typing import List, Optional
import time
import io
from PIL import Image
import os
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    tf = None  # type: ignore
    load_model = None  # type: ignore
import httpx


app = FastAPI(title="Inclusive Classroom Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,allow_headers=["*"]
)


class FaceBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FacePrediction(BaseModel):
    id: int
    box: FaceBox
    emotion: str
    confidence: float
    is_confused: bool


class EmotionResponse(BaseModel):
    faces: List[FacePrediction]


class TeacherState(BaseModel):
    stress_level: float
    fatigue_level: float
    suggestion: str


class AlertCluster(BaseModel):
    message: str
    affected_count: int
    topic: Optional[str] = None


def _load_emotion_model():
    model_path = os.path.join(os.getcwd(), "fer2013_mini_XCEPTION.102-0.66.hdf5")
    if os.path.exists(model_path):
        try:
            if load_model is None:
                print("TensorFlow/Keras not available; skipping model load.")
                return None
            model = load_model(model_path, compile=False)
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    return None


EMOTION_MODEL = _load_emotion_model()
EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Inbuilt Analysis Configuration ---
# Using inbuilt emotion analysis instead of external APIs
print("Using inbuilt emotion analysis system - no external API required")


def _detect_faces(gray_img: np.ndarray) -> List[List[int]]:
    faces = FACE_CASCADE.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return [list(map(int, f)) for f in faces]


def _preprocess_faces_batch(img_arr: np.ndarray, faces: List[List[int]]) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    if len(faces) == 0:
        return None
    processed: List[np.ndarray] = []
    for (x, y, w, h) in faces:
        roi = gray[max(0,y):y+h, max(0,x):x+w]
        if roi.size == 0:
            continue
        face_resized = cv2.resize(roi, (64, 64))
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=-1)
        processed.append(face_resized)
    if not processed:
        return None
    batch = np.stack(processed, axis=0)
    return batch


def _predict_emotions_batch(img_arr: np.ndarray, faces: List[List[int]]) -> List[dict]:
    batch = _preprocess_faces_batch(img_arr, faces)
    if batch is None:
        return []
    # If EMOTION_MODEL is not available, use a lightweight deterministic fallback
    if EMOTION_MODEL is None:
        # Simple heuristic: base emotion on mean pixel intensity in the face ROI
        preds = []
        for face in batch:
            mean_val = float(np.mean(face))
            # map intensity to one of the emotions (deterministic)
            if mean_val > 0.75:
                p = np.array([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
            elif mean_val > 0.6:
                p = np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0])
            elif mean_val > 0.4:
                p = np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0])
            else:
                p = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
            preds.append(p)
        preds = np.stack(preds, axis=0)
    else:
        preds = EMOTION_MODEL.predict(batch, verbose=0)
    results = []
    for i in range(len(faces)):
        probs = preds[i]
        idx = int(np.argmax(probs))
        emotion = EMOTIONS[idx] if idx < len(EMOTIONS) else "neutral"
        confidence = float(probs[idx]) if idx < len(probs) else 0.0
        is_confused = (emotion in ["fear", "sad"]) or (confidence < 0.35)
        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "is_confused": is_confused,
        })
    return results


# --- Simple Centroid Tracker (in-memory) ---
_next_track_id: int = 1
_tracks: dict = {}
_last_seen: dict = {}


def _assign_ids(faces: List[List[int]]) -> List[int]:
    global _next_track_id, _tracks, _last_seen
    now = time.time()
    # expire old tracks
    to_delete = [tid for tid, ts in _last_seen.items() if now - ts > 10.0]
    for tid in to_delete:
        _tracks.pop(tid, None)
        _last_seen.pop(tid, None)
    ids: List[int] = []
    used_tracks = set()
    # compute centroids for detections
    det_centroids = [((x + w // 2), (y + h // 2)) for (x, y, w, h) in faces]
    # greedy match by nearest centroid
    for det_idx, (dx, dy) in enumerate(det_centroids):
        best_tid = None
        best_dist = 1e9
        for tid, (tx, ty) in _tracks.items():
            if tid in used_tracks:
                continue
            dist = (dx - tx) ** 2 + (dy - ty) ** 2
            if dist < best_dist:
                best_dist = dist
                best_tid = tid
        if best_tid is not None and best_dist < (60 ** 2):
            ids.append(best_tid)
            _tracks[best_tid] = (dx, dy)
            _last_seen[best_tid] = now
            used_tracks.add(best_tid)
        else:
            tid = _next_track_id
            _next_track_id += 1
            _tracks[tid] = (dx, dy)
            _last_seen[tid] = now
            ids.append(tid)
    return ids


## removed old simulation path


def _simulate_teacher_state(text_signal: Optional[str], pace: Optional[float]) -> TeacherState:
    base = 0.3
    stress = base
    fatigue = base
    if text_signal:
        lower = text_signal.lower()
        if any(k in lower for k in ["hurry","quick","fast"]):
            stress += 0.3
        if any(k in lower for k in ["tired","exhausted","break"]):
            fatigue += 0.4
    if pace is not None:
        if pace > 180:
            stress += 0.2
        elif pace < 100:
            fatigue += 0.2
    stress = max(0.0, min(1.0, stress))
    fatigue = max(0.0, min(1.0, fatigue))
    suggestion = ""
    if stress > 0.6:
        suggestion = "Consider a 60-second breathing break and slow the pace."
    elif fatigue > 0.6:
        suggestion = "Hydrate and add a short stretch; consider pacing the session."
    else:
        suggestion = "You're maintaining a healthy pace."
    return TeacherState(stress_level=stress, fatigue_level=fatigue, suggestion=suggestion)


class SummarizeRequest(BaseModel):
    transcript: str
    topic: Optional[str] = None


class SummarizeResponse(BaseModel):
    summary: str
    real_life_connections: List[str]


@app.post("/api/emotion", response_model=EmotionResponse)
async def detect_emotion(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    img_arr = np.array(image)
    if EMOTION_MODEL is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    faces = _detect_faces(gray)
    if len(faces) == 0:
        return EmotionResponse(faces=[])
    preds = _predict_emotions_batch(img_arr, faces)
    ids = _assign_ids(faces)
    results: List[FacePrediction] = []
    for (x, y, w, h), pid, p in zip(faces, ids, preds):
        results.append(FacePrediction(
            id=pid,
            box=FaceBox(x=x, y=y, w=w, h=h),
            emotion=p["emotion"],
            confidence=p["confidence"],
            is_confused=p["is_confused"],
        ))
    return EmotionResponse(faces=results)


class CaptionsRequest(BaseModel):
    text: str
    language: str = "en"


class CaptionsResponse(BaseModel):
    captions: List[str]


@app.post("/api/captions", response_model=CaptionsResponse)
async def generate_captions(req: CaptionsRequest):
    chunks = []
    words = req.text.split()
    chunk = []
    for w in words:
        chunk.append(w)
        if len(" ".join(chunk)) > 40:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return CaptionsResponse(captions=chunks)


class TeacherMonitorRequest(BaseModel):
    text_signal: Optional[str] = None
    words_per_minute: Optional[float] = None


@app.post("/api/teacher", response_model=TeacherState)
async def monitor_teacher(req: TeacherMonitorRequest):
    return _simulate_teacher_state(req.text_signal, req.words_per_minute)


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    text = req.transcript.strip()
    if not text:
        return SummarizeResponse(summary="No content to summarize.", real_life_connections=[])
    # Heuristic summary
    sentences = [s.strip() for s in text.replace("\n"," ").split('.') if s.strip()]
    key_points = sentences[:3]
    summary = " ".join(key_points)
    # Simple real-life connections generator
    connections = []
    topic = req.topic or "the lesson"
    connections.append(f"Relate {topic} to daily budgeting or planning.")
    connections.append(f"Find {topic} examples in your home or neighborhood.")
    connections.append(f"Explain {topic} to a friend using a real scenario.")
    return SummarizeResponse(summary=summary, real_life_connections=connections)


class AlertsRequest(BaseModel):
    confused_flags: List[bool]
    topic: Optional[str] = None


@app.post("/api/alerts", response_model=AlertCluster)
async def alerts(req: AlertsRequest):
    count = sum(1 for f in req.confused_flags if f)
    topic = req.topic or "current section"
    if count == 0:
        return AlertCluster(message="No significant confusion detected.", affected_count=0, topic=topic)
    return AlertCluster(
        message=f"{count} students show signs of confusion in {topic} → Suggest a quick pause for clarification.",
        affected_count=count,
        topic=topic,
    )


@app.get("/api/health")
async def health():
    return JSONResponse({"status":"ok"})

@app.get("/api/test-analysis")
async def test_analysis():
    """Test the inbuilt emotion analysis system"""
    test_data = [
        {
            "t": 1234567890,
            "faces": [
                {"id": 1, "emotion": "happy", "confidence": 0.8, "is_confused": False},
                {"id": 1, "emotion": "confused", "confidence": 0.6, "is_confused": True}
            ]
        }
    ]
    
    try:
        analysis = _analyze_emotion_data(test_data)
        return JSONResponse({
            "status": "success", 
            "message": "Inbuilt analysis system working",
            "test_result": analysis
        })
    except Exception as e:
        return JSONResponse({"error": f"Analysis test failed: {e}"}, status_code=500)


class InsightsRequest(BaseModel):
    recent: List[dict]


class InsightsResponse(BaseModel):
    overall: str
    summary: str
    advice: str


def _build_gemini_prompt(recent: List[dict]) -> str:
    counts: dict = {}
    confused = 0
    total_faces = 0
    for snap in recent:
        faces = snap.get("faces", [])
        for f in faces:
            total_faces += 1
            emotion = f.get("emotion", "unknown")
            counts[emotion] = counts.get(emotion, 0) + 1
            if f.get("is_confused"):
                confused += 1
    lines = []
    lines.append("You are a helpful classroom observer.")
    lines.append("Analyze recent student facial-emotion detections and infer how the class is going.")
    lines.append("Return a short JSON with fields: overall, summary, advice.")
    lines.append("overall must be one of: \"great\", \"okay\", \"not_good\".")
    lines.append("Use caution—confidence values are noisy; emphasize trends not single frames.")
    lines.append(f"total_faces={total_faces}")
    lines.append(f"confused_estimate={confused}")
    lines.append("emotion_counts=" + str(counts))
    lines.append("Recent samples:")
    lines.append(str(recent[-8:]))
    return "\n".join(lines)


def _analyze_emotion_data(recent_data):
    """Analyze emotion data and generate insights without external API"""
    if not recent_data:
        return {
            "overall": "unknown",
            "summary": "No data available for analysis",
            "advice": "Start the camera and realtime detection to collect student data"
        }
    
    # Count emotions and confusion
    emotion_counts = {}
    confused_count = 0
    total_detections = 0
    confidence_sum = 0
    recent_emotions = []
    
    for detection in recent_data:
        faces = detection.get("faces", [])
        for face in faces:
            total_detections += 1
            emotion = face.get("emotion", "neutral")
            confidence = face.get("confidence", 0)
            is_confused = face.get("is_confused", False)
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_sum += confidence
            recent_emotions.append(emotion)
            
            if is_confused:
                confused_count += 1
    
    if total_detections == 0:
        return {
            "overall": "unknown",
            "summary": "No student detected",
            "advice": "Ensure student is visible in camera frame"
        }
    
    # Calculate metrics
    avg_confidence = confidence_sum / total_detections
    confusion_rate = confused_count / total_detections
    
    # Determine overall state
    if confusion_rate > 0.4:
        overall = "not_good"
    elif confusion_rate > 0.2 or avg_confidence < 0.4:
        overall = "okay"
    else:
        overall = "great"
    
    # Generate summary based on data
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    summary_parts = []
    
    if confusion_rate > 0.3:
        summary_parts.append(f"Student shows confusion in {confused_count}/{total_detections} detections")
    if dominant_emotion in ["fear", "sad"]:
        summary_parts.append(f"Student appears distressed ({dominant_emotion})")
    elif dominant_emotion == "happy":
        summary_parts.append("Student appears engaged and positive")
    elif dominant_emotion == "neutral":
        summary_parts.append("Student appears focused and neutral")
    
    if avg_confidence < 0.4:
        summary_parts.append("Low confidence in emotion detection - student may be distracted")
    
    summary = ". ".join(summary_parts) if summary_parts else f"Student shows {dominant_emotion} emotion with {avg_confidence:.1%} average confidence"
    
    # Generate advice based on analysis
    advice_parts = []
    
    if confusion_rate > 0.4:
        advice_parts.append("Student needs immediate attention - approach individually and offer clarification")
    elif confusion_rate > 0.2:
        advice_parts.append("Monitor student closely - consider checking if they need help")
    
    if dominant_emotion in ["fear", "sad"]:
        advice_parts.append("Student appears distressed - offer support and check if they're okay")
    elif dominant_emotion == "angry":
        advice_parts.append("Student shows frustration - consider breaking down the task or offering assistance")
    elif dominant_emotion == "happy":
        advice_parts.append("Student is responding well - continue current approach")
    
    if avg_confidence < 0.4:
        advice_parts.append("Student may be distracted - try to re-engage with direct questions")
    
    advice = ". ".join(advice_parts) if advice_parts else "Continue monitoring student behavior and provide support as needed"
    
    return {
        "overall": overall,
        "summary": summary,
        "advice": advice,
        "stats": {
            "total_detections": total_detections,
            "confusion_rate": f"{confusion_rate:.1%}",
            "avg_confidence": f"{avg_confidence:.1%}",
            "dominant_emotion": dominant_emotion,
            "emotion_breakdown": emotion_counts
        }
    }

@app.post("/api/insights", response_model=InsightsResponse)
async def insights(req: InsightsRequest):
    """Generate insights using inbuilt analysis instead of Gemini API"""
    try:
        analysis = _analyze_emotion_data(req.recent or [])
        return InsightsResponse(
            overall=analysis["overall"],
            summary=analysis["summary"],
            advice=analysis["advice"]
        )
    except Exception as e:
        return JSONResponse({"error": f"Analysis failed: {e}"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)


