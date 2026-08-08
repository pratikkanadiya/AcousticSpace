"""
AcousticSpace inference API.

Run with:

    uvicorn main:app --reload --port 8000

Environment variables (optional):

    CHECKPOINT_PATH
    ACOUSTIC_STATS_PATH
    ALLOWED_ORIGINS
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import AcousticSpaceModel
from schemas import AnalysisResponse, HealthResponse

from database import (init_db, save_analysis, get_history, delete_history, clear_history)

# ============================================================
# CONFIGURATION
# ============================================================

# Use raw strings for Windows paths.
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    r"D:\InfoTect\model\model_weights\final_multimodal_acoustic_net.pth"
)

ACOUSTIC_STATS_PATH = os.environ.get(
    "ACOUSTIC_STATS_PATH",
    r"D:\InfoTect\model\model_weights\acoustic_stats.json"
)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


# Maximum upload size = 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# Supported audio formats
ALLOWED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
}


# ============================================================
# MODEL HOLDER
# ============================================================

model_holder = {
    "model": None,
    "load_error": None,
}

init_db()

# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting AcousticSpace API...")
    print("=" * 60)

    try:

        print(f"Checkpoint: {CHECKPOINT_PATH}")
        print(f"Acoustic stats: {ACOUSTIC_STATS_PATH}")

        model_holder["model"] = AcousticSpaceModel(
            checkpoint_path=CHECKPOINT_PATH,
            acoustic_stats_path=ACOUSTIC_STATS_PATH,
        )

        model_holder["load_error"] = None

        print(
            f"Model loaded successfully on "
            f"{model_holder['model'].device}"
        )

    except Exception as e:

        model_holder["model"] = None
        model_holder["load_error"] = str(e)

        print(
            f"WARNING: Model failed to load:\n{e}"
        )

    yield

    # ========================================================
    # SHUTDOWN
    # ========================================================

    print("Shutting down AcousticSpace API...")

    model_holder["model"] = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AcousticSpace API",
    description=(
        "Deepfake audio detection via room impulse response "
        "and breathing-pattern mismatch."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    model = model_holder["model"]

    if model is not None:

        return HealthResponse(
            status="ok",
            device=str(model.device),
            model_loaded=True,
            acoustic_stats_loaded=True,
        )

    return HealthResponse(
        status="model_not_loaded",
        device="n/a",
        model_loaded=False,
        acoustic_stats_loaded=False,
    )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):

    model = model_holder["model"]

    # ---------------------------------------------------------
    # Check model
    # ---------------------------------------------------------

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {model_holder['load_error']}"
        )

    # ---------------------------------------------------------
    # Validate extension
    # ---------------------------------------------------------

    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            )
        )

    # ---------------------------------------------------------
    # Read upload
    # ---------------------------------------------------------

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 50MB limit."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # ---------------------------------------------------------
    # Temporary file
    # ---------------------------------------------------------

    tmp_dir = tempfile.mkdtemp(
        prefix="acousticspace_upload_"
    )

    tmp_path = os.path.join(
        tmp_dir,
        f"upload{ext}"
    )

    try:

        with open(tmp_path, "wb") as f:
            f.write(contents)

        print("\n" + "=" * 70)
        print("ACOUSTICSPACE ANALYSIS")
        print("=" * 70)
        print(f"Original filename : {file.filename}")
        print(f"Temporary file   : {tmp_path}")
        print(f"File extension    : {ext}")
        print(f"File size        : {len(contents) / 1024:.2f} KB")
        print(f"Device            : {model.device}")
        print("-" * 70)

        # -----------------------------------------------------
        # Run inference
        # -----------------------------------------------------

        try:

            result = model.analyze_file(
                tmp_path,
                original_filename=file.filename
            )

            print("-" * 70)
            print("ANALYSIS COMPLETED")
            print(f"Verdict           : {result['verdict']}")
            print(
                f"Spoof probability: "
                f"{result['spoof_probability']:.6f}"
            )
            print(
                f"Confidence        : "
                f"{result['confidence_pct']:.2f}%"
            )
            print(
                f"Segments          : "
                f"{result['num_segments']}"
            )
            print("=" * 70 + "\n")

        except ValueError as e:

            print(
                f"[AcousticSpace] Validation/preprocessing error: {e}"
            )

            raise HTTPException(
                status_code=422,
                detail=str(e)
            )

        except Exception as e:

            import traceback

            print("\n" + "=" * 70)
            print("ACOUSTICSPACE INFERENCE ERROR")
            print("=" * 70)

            traceback.print_exc()

            print("=" * 70 + "\n")

            raise HTTPException(
                status_code=500,
                detail=f"Audio analysis failed: {str(e)}"
            )

        result = model.analyze_file(
            tmp_path,
            original_filename=file.filename
        )

        save_analysis(result)
        
        return AnalysisResponse(**result)

    finally:

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True
        )


@app.get("/history")
def history():
    return get_history()

@app.delete("/history/{history_id}")
def remove_history(history_id: int):

    deleted = delete_history(history_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="History record not found."
        )

    return {
        "message": "History deleted successfully"
    }

@app.delete("/history")
def remove_all_history():

    clear_history()

    return {
        "message": "All history deleted successfully"
    }



# ============================================================
# LOCAL DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
