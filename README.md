## Inclusive Classroom Assistant (Emotion-Aware, Accessibility, Insights)

A FastAPI backend plus a lightweight React front-end (CDN) that demonstrates:
- Student emotion and doubt detection
- Teacher well-being monitoring and balanced alerts
- Live captions (chunked transcription input)
- Session summarization with real-life connections

### Prerequisites
- Python 3.10+
- Node (optional) if you prefer running a static server with `npx` (otherwise use Python's builtin server)

### Setup

```bash
# From project root
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Backend (FastAPI)

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `http://localhost:8000/api/health`

### Run Frontend
Two options:

1) Using Python http.server (no Node needed):
```bash
# In project root
python -m http.server 5500
# then open http://localhost:5500/frontend/index.html
```

2) Using npx serve (Node required):
```bash
# In project root
npx --yes serve frontend -l 5500
# then open http://localhost:5500
```

The frontend expects the backend at `http://localhost:8000` (CORS is allowed).

### Integrating the Real Emotion Model (Optional)
- Place your Keras model file at project root (already present: `fer2013_mini_XCEPTION.102-0.66.hdf5`).
- Replace `_simulate_emotion` and `_load_emotion_model` in `backend/app.py` with actual preprocessing + model inference. You will likely need TensorFlow/Keras:
  - Add to `requirements.txt`: `tensorflow` (or `tensorflow-cpu`) and `keras`
  - Preprocess frames to the input size required by mini-XCEPTION (e.g. grayscale 64x64)
  - Run `model.predict` and map logits to emotion labels

### Using the Provided Model in Real Time
- Already wired: backend loads `fer2013_mini_XCEPTION.102-0.66.hdf5` if present and will prefer live inference over simulation.
- Frontend: Click `Start Camera`, then `Start Realtime` to continuously send frames to `/api/emotion` (~0.6s interval) and display predictions.
- If no face is detected, we fallback to center-crop.
- Confusion flag is set when emotion is `fear`/`sad` or confidence is low.

### Troubleshooting
- If TensorFlow fails to import on CPU-only machines, ensure `tensorflow-cpu` is installed (we pinned it in `requirements.txt`).
- If model fails to load, check the exact filename at project root and that it is a valid Keras HDF5 model.
- If camera permission is blocked, allow access in your browser.

### Security & Privacy
- This demo persists no data or audio/video streams.
- For production, use on-device inference or a secure backend with consent management.

### Notes
- This is a self-contained demo optimized for simplicity and clarity.
- Styling uses animated conic/radial gradients for a modern, accessible experience.



