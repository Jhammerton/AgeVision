"""FastAPI serving layer for AgeVision."""
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.config import MODEL_DIR, PROJECT_ROOT
from src.face_detection import FaceDetectionError, crop_detected_face
from src.predict import AgePredictor
from src.validation import MAX_UPLOAD_BYTES, validate_image


def model_checkpoint_path() -> Path:
    """Return the checkpoint configured for the serving process."""
    configured = os.getenv("AGEVISION_CHECKPOINT")
    return Path(configured) if configured else MODEL_DIR / "agevision_regression.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = model_checkpoint_path()
    app.state.predictor = AgePredictor(checkpoint) if checkpoint.exists() else None
    yield

app = FastAPI(title="AgeVision", version="0.1.0", lifespan=lifespan)

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PROJECT_ROOT / "src" / "static" / "index.html")

@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.predictor is not None}

@app.post("/api/v1/predict")
async def predict(request: Request, file: Annotated[UploadFile, File()]):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB upload limit")
    try:
        source = Image.open(BytesIO(content))
        validate_image(source, len(content))
        image = Image.open(BytesIO(content)).convert("RGB")
        face = crop_detected_face(image)
    except (FaceDetectionError, ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if request.app.state.predictor is None:
        raise HTTPException(503, "Model checkpoint is not installed")
    return request.app.state.predictor.predict(face)
