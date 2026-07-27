"""
AcousticSpace — Preprocessing Pipeline
========================================
Deepfake detection via Room Impulse Response (RIR) analysis.

NOTE: This version deliberately avoids `librosa` because librosa depends on
`numba`, and numba's compiled DLLs get blocked by some Windows security
policies (Smart App Control / Application Control), causing an
"ImportError: DLL load failed ... Application Control policy has blocked
this file" error that's outside our control to fix via pip. Everything here
is built from `soundfile` + `numpy` + `scipy` only, which are pure-Python /
standard compiled wheels with no such issue.

Pipeline stages:
 1. Audio ingestion & standardization (load, resample, mono, normalize)
 2. Signal cleaning (DC offset removal, amplitude normalization, clip detection)
 3. Voice Activity Detection (VAD) — isolate speech segments
 4. Segmentation / framing (fixed-length windows for RIR estimation)
 5. Noise floor estimation
 6. RIR-relevant feature extraction (RT60, DRR estimate, log-mel spectrogram)
 7. Final normalization / fixed-length packaging for model input

Usage:
    python preprocessing.py --input path/to/audio_or_dir --output path/to/out_dir
"""

import os
import glob
import json
import argparse
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft
from scipy.stats import linregress
from math import gcd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TARGET_SR = 16000          # standard for speech/RIR analysis
FRAME_MS = 30              # VAD / noise-floor analysis frame size
SEGMENT_SEC = 2.0          # fixed-length segment for feature extraction
N_MELS = 64
CLIP_THRESHOLD = 0.999     # peak amplitude fraction considered "clipped"
N_FFT = 512
HOP_LENGTH = 160           # 10ms at 16kHz


# ----------------------------------------------------------------------
# 1. Audio ingestion & standardization
# ----------------------------------------------------------------------
def load_and_standardize(path, target_sr=TARGET_SR):
    """Load audio, force mono, resample to target_sr, return float32 in [-1, 1]."""
    audio, sr = sf.read(path, always_2d=True)  # shape: (n_samples, n_channels)

    # Downmix to mono
    audio = np.mean(audio, axis=1).astype(np.float32)

    # Resample if needed using polyphase resampling (no compiled extras needed)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)
        sr = target_sr

    return audio, sr


# ----------------------------------------------------------------------
# 2. Signal cleaning
# ----------------------------------------------------------------------
def remove_dc_offset(audio):
    return audio - np.mean(audio)


def detect_clipping(audio, threshold=CLIP_THRESHOLD):
    """Return fraction of samples near full-scale (potential clipping)."""
    clipped = np.sum(np.abs(audio) >= threshold)
    return clipped / max(len(audio), 1)


def normalize_amplitude(audio, method="peak", target_dbfs=-3.0):
    """Peak or RMS normalization."""
    if method == "peak":
        peak = np.max(np.abs(audio)) + 1e-9
        gain = (10 ** (target_dbfs / 20)) / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        target_rms = 10 ** (target_dbfs / 20)
        gain = target_rms / rms
    else:
        raise ValueError("method must be 'peak' or 'rms'")
    return audio * gain


def clean_signal(audio):
    audio = remove_dc_offset(audio)
    clip_frac = detect_clipping(audio)
    audio = normalize_amplitude(audio, method="peak", target_dbfs=-3.0)
    return audio, {"clipping_fraction": float(clip_frac)}


# ----------------------------------------------------------------------
# 3. Voice Activity Detection (VAD) — simple energy-based, dependency-free
# ----------------------------------------------------------------------
def run_vad(audio, sr=TARGET_SR, top_db=30, frame_ms=FRAME_MS):
    """
    Energy-based VAD: computes short-time RMS energy per frame, converts to
    dB relative to the loudest frame, and keeps frames within `top_db` of
    that peak as "speech". Merges consecutive speech frames into regions.
    Returns list of (start_sample, end_sample).
    """
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    if frame_len < 1 or len(audio) < frame_len:
        return [(0, len(audio))]

    n_frames = 1 + (len(audio) - frame_len) // hop
    energies = np.empty(n_frames)
    starts = np.empty(n_frames, dtype=int)
    for i in range(n_frames):
        s = i * hop
        frame = audio[s:s + frame_len]
        energies[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)
        starts[i] = s

    ref = np.max(energies) + 1e-12
    db = 20 * np.log10(energies / ref + 1e-12)
    is_speech = db > -top_db

    regions = []
    cur_start = None
    for i, flag in enumerate(is_speech):
        if flag and cur_start is None:
            cur_start = starts[i]
        elif not flag and cur_start is not None:
            regions.append((int(cur_start), int(starts[i] + frame_len)))
            cur_start = None
    if cur_start is not None:
        regions.append((int(cur_start), len(audio)))

    if not regions:
        regions = [(0, len(audio))]
    return regions


def extract_speech_audio(audio, regions, sr=TARGET_SR, min_region_sec=0.3):
    """Concatenate speech regions above a minimum duration."""
    min_len = int(min_region_sec * sr)
    chunks = [audio[s:e] for s, e in regions if (e - s) >= min_len]
    if not chunks:
        return audio  # fallback: no confident VAD regions, keep original
    return np.concatenate(chunks)


# ----------------------------------------------------------------------
# 4. Segmentation / framing
# ----------------------------------------------------------------------
def segment_audio(audio, sr=TARGET_SR, segment_sec=SEGMENT_SEC, overlap=0.5):
    """Split into fixed-length, tapered, overlapping segments."""
    seg_len = int(segment_sec * sr)
    hop = int(seg_len * (1 - overlap))
    window = np.hanning(seg_len)

    segments = []
    for start in range(0, max(len(audio) - seg_len, 0) + 1, hop):
        seg = audio[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg * window)

    # Handle audio shorter than one segment
    if not segments:
        seg = np.pad(audio, (0, max(seg_len - len(audio), 0)))[:seg_len]
        segments.append(seg * window)

    return np.stack(segments)  # shape: (n_segments, seg_len)


# ----------------------------------------------------------------------
# 5. Noise floor estimation
# ----------------------------------------------------------------------
def estimate_noise_floor(audio, sr=TARGET_SR, frame_ms=30):
    """
    Estimate noise floor using the lowest-energy percentile of frames
    (assumes noise-only frames exist, e.g. silence/background between speech).
    """
    frame_len = int(sr * frame_ms / 1000)
    energies = [
        np.mean(audio[i:i + frame_len] ** 2)
        for i in range(0, len(audio) - frame_len + 1, frame_len)
    ]
    if not energies:
        return -np.inf
    energies = np.array(energies)
    noise_energy = np.percentile(energies, 10)  # bottom 10% = likely noise
    noise_dbfs = 10 * np.log10(noise_energy + 1e-12)
    return float(noise_dbfs)


# ----------------------------------------------------------------------
# 6. RIR-relevant feature extraction
# ----------------------------------------------------------------------
def estimate_rt60(segment, sr=TARGET_SR):
    """
    Rough RT60 estimate via Schroeder backward-integration on the segment's
    energy decay curve. This is a coarse blind estimate (proper RIR-based
    RT60 requires an actual impulse response, e.g. from a sine sweep) —
    here we approximate using the decay of the segment envelope, which is
    a common lightweight proxy for detecting acoustic-environment mismatch.
    """
    energy = segment ** 2
    if np.sum(energy) < 1e-9:
        return 0.0

    # Schroeder integration: reverse cumulative sum of energy
    sch = np.cumsum(energy[::-1])[::-1]
    sch_db = 10 * np.log10(sch / (np.max(sch) + 1e-12) + 1e-12)

    # Fit slope between -5dB and -25dB (standard RT20->RT60 extrapolation)
    t = np.arange(len(sch_db)) / sr
    mask = (sch_db <= -5) & (sch_db >= -25)
    if np.sum(mask) < 2:
        return 0.0

    slope, intercept, _, _, _ = linregress(t[mask], sch_db[mask])
    if slope == 0:
        return 0.0
    rt60 = -60 / slope
    return float(np.clip(rt60, 0, 3.0))  # clip to plausible room range


def estimate_drr(segment, sr=TARGET_SR, direct_window_ms=5):
    """
    Rough Direct-to-Reverberant Ratio proxy: energy in the first
    `direct_window_ms` after the segment's peak vs. remaining energy.
    """
    if len(segment) == 0 or np.max(np.abs(segment)) < 1e-9:
        return 0.0
    peak_idx = np.argmax(np.abs(segment))
    win = int(sr * direct_window_ms / 1000)
    direct = segment[peak_idx:peak_idx + win]
    reverb = np.concatenate([segment[:peak_idx], segment[peak_idx + win:]])

    direct_energy = np.sum(direct ** 2) + 1e-12
    reverb_energy = np.sum(reverb ** 2) + 1e-12
    drr_db = 10 * np.log10(direct_energy / reverb_energy)
    return float(drr_db)


def _hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def _mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def _mel_filterbank(sr, n_fft, n_mels):
    """Build a triangular mel filterbank matrix, shape (n_mels, n_fft//2 + 1)."""
    n_freqs = n_fft // 2 + 1
    fmin, fmax = 0.0, sr / 2.0
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    fbank = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_freqs:
                fbank[m - 1, k] = (right - k) / (right - center)
    return fbank


_MEL_FB_CACHE = {}


def extract_log_mel(segment, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Log-mel spectrogram computed via scipy.signal.stft + a manual mel filterbank
    (no librosa/numba dependency)."""
    key = (sr, n_mels, n_fft)
    if key not in _MEL_FB_CACHE:
        _MEL_FB_CACHE[key] = _mel_filterbank(sr, n_fft, n_mels)
    fbank = _MEL_FB_CACHE[key]

    _, _, Zxx = stft(segment, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    power = np.abs(Zxx) ** 2  # shape: (n_freqs, n_frames)

    mel = fbank @ power  # (n_mels, n_frames)
    log_mel = 10 * np.log10(mel + 1e-10)
    # normalize relative to peak, like librosa's power_to_db(ref=np.max)
    log_mel = log_mel - np.max(log_mel)
    return log_mel


def extract_features_per_segment(segments, sr=TARGET_SR):
    features = []
    for seg in segments:
        feat = {
            "rt60_est": estimate_rt60(seg, sr),
            "drr_est_db": estimate_drr(seg, sr),
            "log_mel": extract_log_mel(seg, sr).tolist(),
        }
        features.append(feat)
    return features


# ----------------------------------------------------------------------
# 7. Full pipeline
# ----------------------------------------------------------------------
def preprocess_file(path, sr=TARGET_SR, save_segments_dir=None):
    """Run the full preprocessing pipeline on a single audio file."""
    audio, sr = load_and_standardize(path, target_sr=sr)
    audio, clean_meta = clean_signal(audio)

    vad_regions = run_vad(audio, sr=sr)
    speech_audio = extract_speech_audio(audio, vad_regions, sr=sr)

    noise_dbfs = estimate_noise_floor(speech_audio, sr=sr)

    segments = segment_audio(speech_audio, sr=sr)
    features = extract_features_per_segment(segments, sr=sr)

    if save_segments_dir:
        os.makedirs(save_segments_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        for i, seg in enumerate(segments):
            sf.write(os.path.join(save_segments_dir, f"{base}_seg{i:03d}.wav"), seg, sr)

    result = {
        "file": os.path.basename(path),
        "sample_rate": sr,
        "duration_sec": float(len(audio) / sr),
        "clipping_fraction": clean_meta["clipping_fraction"],
        "num_vad_regions": len(vad_regions),
        "speech_duration_sec": float(len(speech_audio) / sr),
        "noise_floor_dbfs": noise_dbfs,
        "num_segments": len(segments),
        "segment_features": features,
    }
    return result


def preprocess_directory(input_dir, output_dir, save_segments=False):
    os.makedirs(output_dir, exist_ok=True)
    audio_exts = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")
    files = []
    for ext in audio_exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    results = []
    for f in sorted(files):
        try:
            seg_dir = os.path.join(output_dir, "segments") if save_segments else None
            res = preprocess_file(f, save_segments_dir=seg_dir)
            results.append(res)
            print(f"[OK] {os.path.basename(f)} -> {res['num_segments']} segments, "
                  f"RT60(avg)={np.mean([s['rt60_est'] for s in res['segment_features']]):.2f}s")
        except Exception as e:
            print(f"[FAIL] {f}: {e}")

    out_json = os.path.join(output_dir, "preprocessing_report.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved report -> {out_json}")
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcousticSpace preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Audio file or directory of audio files")
    parser.add_argument("--output", required=True, help="Output directory for report/segments")
    parser.add_argument("--save-segments", action="store_true", help="Save segmented WAV files")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        preprocess_directory(args.input, args.output, save_segments=args.save_segments)
    else:
        os.makedirs(args.output, exist_ok=True)
        seg_dir = os.path.join(args.output, "segments") if args.save_segments else None
        res = preprocess_file(args.input, save_segments_dir=seg_dir)
        with open(os.path.join(args.output, "preprocessing_report.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k != "segment_features"}, indent=2))
