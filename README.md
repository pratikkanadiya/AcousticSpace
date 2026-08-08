# AcousticSpace

AcousticSpace is a deepfake / synthetic-speech detector that flags spoofed audio using room acoustics and breathing-pattern artifacts rather than pure spectral cues. 

The idea: synthesized or replayed speech rarely reproduces a physically consistent room impulse response or natural breathing behavior, so a multi-modal model trained on ASVspoof-style data can catch what waveform-only detectors miss.


# How it works

(1) Preprocessing (preprocess.py) 
  
turns a raw audio file into:

  --> a log-mel spectrogram per 5-second segment

  --> a room impulse response descriptor estimated via LPC-residual autocorrelation

  --> 9 acoustic features: room energy, clarity (C50), peak index, spectral centroid, DRR, RT60, and breathing stats (count, avg/std interval between breath events detected from low-frequency, low-energy, high-ZCR frames)

(2) MultiModalAcousticNet (model.py)

fuses three branches:

  —-> a 2D CNN over the spectrogram, a 1D CNN over the room descriptor, and an MLP over the 9 acoustic features 
  
  —-> into a fused vector classified as bonafide vs spoof.

(3) ASVspoofDataset (dataset.py)

loads preprocessed segment reports + ASVspoof protocol files, computes/normalizes acoustic feature stats, and serves (spectrogram, room_descriptor, acoustic_features, label) tuples.

(4) train.py

runs the full pipeline: preprocess raw .flac files if needed, train with AMP + torch.compile, checkpoint/resume, and save the best/final model plus acoustic_stats.json.

(5) server/inference.py

loads the trained checkpoint and re-runs the same preprocess.py pipeline at request time to produce a verdict.

(6) server/main.py

exposes this over a FastAPI /analyze endpoint; results are logged to SQLite (server/database.py) and browsable via /history.

(7) acousticspace-dashboard

is a React SPA that uploads a file, renders the waveform (wavesurfer.js), and displays the verdict, confidence, and acoustic metrics.

# Repository layout

```
.
├── model.py                # MultiModalAcousticNet (spec CNN + room CNN + acoustic MLP + classifier head)
├── dataset.py              # ASVspoofDataset — loads protocol + preprocessed segment reports
├── train.py                # End-to-end training script (preprocess -> train -> checkpoint)
├── server/
|   |── preprocess.py       # Audio -> spectrogram / room descriptor / acoustic feature extraction
|   |── model.py            # MultiModalAcousticNet (spec CNN + room CNN + acoustic MLP + classifier head)
│   ├── main.py             # FastAPI app: /health, /analyze, /history endpoints
│   ├── inference.py        # AcousticSpaceModel — loads checkpoint, runs inference, repairs bad stats
│   ├── database.py         # SQLite history storage (analysis_history table)
│   ├── schemas.py          # Pydantic request/response models
│   ├── acousticspace.db    # SQLite database file
|   ├── Dockerfile          # backend container configuration environment layout
└── acousticspace-dashboard/
    ├── src/App.jsx          # Main UI: upload, waveform, verdict, metrics, history
    ├── src/main.jsx
    ├── src/index.css
    ├── vite.config.js
    |── package.json
    ├── Dockerfile           # Add frontend server container configuration asset layout

```

## Model Architecture

AcousticSpace uses a multimodal architecture that combines spectral, room-acoustic, and acoustic-feature information to detect synthetic/deepfake audio.

----------------------------------------------------------------------------------------------------------------------------
| Branch             | Input                               | Architecture                                    | Output Dim  |
|--------------------|-------------------------------------|-------------------------------------------------|-------------|
| **SpecBranch**     | Log-Mel spectrogram `(1 × 128 × T)` | 5 × `(Conv2D + BatchNorm + ReLU + MaxPool)`     | **512**     |
| **RoomBranch**     | Room descriptor `(1 × L)`           | 3 × `(Conv1D k=7 + BatchNorm + ReLU + MaxPool)` | **128**     |
| **AcousticBranch** | 9 normalized acoustic features      | 2-layer MLP `(64 → 32)`                         | **32**      |
----------------------------------------------------------------------------------------------------------------------------

### Multimodal Feature Fusion

The three branches extract complementary information:

- **SpecBranch** captures spectral and vocal characteristics from the Log-Mel spectrogram.
- **RoomBranch** captures room/environmental characteristics from the room descriptor.
- **AcousticBranch** captures handcrafted acoustic features such as RT60, DRR, room clarity, spectral centroid, and breathing-related features.

The extracted representations are then combined before the final classification layer:

```text
                    Audio
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
     SpecBranch   RoomBranch  AcousticBranch
          │           │           │
        512          128          32
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
              Feature Fusion
                      │
                      ▼
                Classifier
                      │
                 ┌────┴────┐
                 ▼         ▼
             Bonafide    Spoof
```

The three branch outputs (512 + 128 + 32 = 672) are concatenated and passed through a ClassifierHead (256 → 64 → 2) with dropout and BatchNorm to produce bonafide/spoof logits.

# Acoustic feature vector

```
0 room_energy
1 room_clarity_c50
2 room_peak_index
3 room_centroid
4 drr_db
5 rt60_sec
6 breath_count
7 avg_breath_interval
8 breath_variation (std_interval)
```

## Fast API Endpoints

---------------------------------------------------------------------------------------------------------------------------------------------------------------
| Method   | Path       | Description                                                                                                                          |
|----------|------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `GET`    | `/health`  | Returns model load status and device information.                                                                                    |
| `POST`   | `/analyze` | Upload an audio file (`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, ≤ 50 MB). Returns verdict, confidence, spoof probability, per-segment predictions, and acoustic summary.                                                                                                                             |
| `GET`    | `/history` | Lists past audio analyses stored in SQLite.                                                                                          |
| `DELETE` | `/history/{id}` | Deletes a specific history entry by ID.                                                                                         |
| `DELETE` | `/history` | Clears all stored analysis history.                                                                                                  |
----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Supported Audio Formats

```
.wav
.flac
.mp3
.ogg
.m4a
```

## Multi-Service Container Breakdown

The application splits the client-side user interface and the heavy machine learning inference engine into two separate, optimized containers.

### 1. Backend Service (`server`)
The backend container encapsulates the complete Python machine learning execution runtime.
* **Base Image**: `python:3.11-slim` (Minimal Linux footprint optimized for faster build times and deployment safety)
* **Application Server**: `Uvicorn` running a `FastAPI` instance.
* **Network Exposure**: Internal port `8000` mapped directly to host port `8000`.
* **Execution Boundary**: Bound to the host network on `0.0.0.0` to permit cross-container connection requests originating from the client application.
* **Persistence & Volumes**: Mounts local configurations and `./server/model_weights/` directly into the runtime environment to securely load the network tracking state on startup.

### 2. Frontend Service (`acousticspace-dashboard`)
The frontend container hosts the client-side visual dashboard asset compilation engine.
* **Base Image**: `node:20-alpine` (Ultra-lightweight Linux distribution explicitly tailored for fast node module execution)
* **Dev Server**: `Vite` ecosystem managing rapid asset bundling.
* **Network Exposure**: Internal port `5173` mapped to host port `5173`.
* **Cross-Origin Configuration**: Uses the `--host` compilation flag to make the live UI fully accessible outside the local loopback boundary.
* **Persistence & Volumes**: Mounts the local working workspace onto the running container directory (`/app`) while isolating the heavy containerized `/app/node_modules/` to support instantaneous hot-reloading (HMR) when local code edits occur.

# Project Development Timeline

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Week         | Tasks                                                                                                                                               |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Week 1**   | Find and select the dataset; build the audio preprocessing pipeline.                                                                                |
| **Week 2**   | Assign labels and standardize the dataset; build the deep learning model; develop the training pipeline.                                            |
| **Week 3**   | Preprocess the complete dataset (~50K audio files); train the CNN model on the processed data                                                       |
| **Week 4**   | build the React dashboard, Create the FastAPI backend; test the model on various audio files; Dockerize the complete AcousticSpace application.     |
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
