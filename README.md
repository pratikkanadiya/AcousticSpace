# AcousticSpace — Deepfake Detection via Room Impulse Response (RIR)

## What is this project?

Every room "colors" sound in its own way — reflections off walls, absorption
from furniture, size of the space. This acoustic signature is called the
**Room Impulse Response (RIR)**.

Real recordings naturally carry their room's RIR signature. AI-generated
(deepfake/TTS/voice-cloned) speech is often synthesized without a real room,
or has reverb added artificially/inconsistently, or is spliced from clips
recorded in different rooms.

**Core idea:** analyze the room-acoustic fingerprint in a piece of audio.
If it looks unnatural, inconsistent, or doesn't match the claimed
environment, flag the audio as potentially fake or manipulated.

## Project pipeline (3 phases)

```
Raw audio  ─▶  1. PREPROCESSING  ─▶  2. FEATURE EXTRACTION  ─▶  3. CLASSIFICATION
              (clean, segment,        (RT60, DRR, spectral       (real vs fake /
               VAD)                    RIR descriptors)           tampered)
```

| Phase | Status | Folder |
|---|---|---|
| 1. Preprocessing | ✅ Done | `src/preprocessing.py` |
| 2. Feature extraction | ⚙️ built into preprocessing (basic) | `src/preprocessing.py` |
| 3. Classifier | 🔲 Not started | `src/` (to be added) |

## Folder structure

```
AcousticSpace/
├── data/
│   ├── raw/          # original audio files you collect (real + fake samples)
│   ├── processed/    # cleaned/normalized audio after preprocessing
│   └── segments/      # fixed-length audio chunks + extracted features
├── src/
│   └── preprocessing.py   # the preprocessing pipeline (load, clean, VAD, segment, features)
├── notebooks/         # exploration / experiments (Jupyter, optional)
├── models/            # trained classifier files go here (later phase)
├── reports/            # JSON/CSV outputs, evaluation results, plots
├── tests/              # test scripts
└── README.md
```

## How to run what exists so far

```bash
# from inside AcousticSpace/
pip install -r requirements.txt

# preprocess a single file
python src/preprocessing.py --input data/raw/sample.wav --output reports/ --save-segments

# preprocess a whole folder of audio
python src/preprocessing.py --input data/raw/ --output reports/ --save-segments
```

This produces:
- Segmented WAV clips in `reports/segments/`
- A `preprocessing_report.json` with per-file and per-segment stats:
  RT60 estimate, DRR estimate, log-mel spectrogram, noise floor, etc.

## What you need to do next (in order)

1. **Collect data** — put some real recordings and some AI-generated/deepfake
   audio samples into `data/raw/`. You'll need both classes to train a
   detector. (I can help you find/build a dataset plan.)
2. **Run preprocessing** on all files to generate consistent segments + features.
3. **Build a classifier** (Phase 3) — a simple model (e.g. logistic regression
   or small neural net) that takes the extracted features and predicts
   real vs. fake. I haven't built this yet — happy to do it next.
4. **Evaluate** — accuracy, precision/recall, confusion matrix on held-out data.

## Key terms (glossary for beginners)

- **RIR (Room Impulse Response):** the "fingerprint" of how a room reflects/absorbs sound.
- **RT60:** time it takes for sound to decay by 60dB in a room — bigger/harder rooms = longer RT60.
- **DRR (Direct-to-Reverberant Ratio):** how much of the sound is "direct" (straight from source to mic) vs. "reverberant" (bounced off walls first).
- **VAD (Voice Activity Detection):** automatically detecting which parts of audio contain speech vs. silence/noise.
- **Blind RIR estimation:** estimating the room's acoustic fingerprint *without* having a reference recording — just from the speech itself. Harder, but realistic for detecting deepfakes in the wild.
