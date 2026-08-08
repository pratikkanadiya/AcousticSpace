"""
AcousticSpace - production inference pipeline.

Emergency / no-retraining repair:
1. Loads the existing MultiModalAcousticNet checkpoint.
2. Removes DataParallel "module." prefixes.
3. Repairs corrupted BatchNorm running_mean/running_var buffers in memory.
   The supplied checkpoint was found to contain 8 non-finite BatchNorm buffers.
4. Repairs invalid acoustic normalization statistics in memory.
   The supplied acoustic_stats.json has NaN in the 9th feature
   (breath_variation/std_interval), so that feature falls back to mean=0,
   std=1 instead of poisoning the model input.
5. Reuses preprocess.py -> audio_to_spectrogram() exactly.
6. Produces finite probabilities, verdict and confidence.
"""

import os
import sys
import json
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import librosa


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT MODULES
# ============================================================

from model import MultiModalAcousticNet
from preprocess import audio_to_spectrogram


class AcousticSpaceModel:
    """
    AcousticSpace inference pipeline.

    Inputs:
        - Mel spectrogram
        - Room descriptor
        - 9 acoustic features

    Output:
        - Authentic / synthetic verdict
        - Spoof probability
        - Confidence
        - Acoustic summary
        - Segment predictions
    """

    def __init__(
        self,
        checkpoint_path: str,
        acoustic_stats_path: str,
        device: str = None,
    ):
        # ========================================================
        # DEVICE
        # ========================================================

        self.device = torch.device(
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        print(f"[AcousticSpace] Device: {self.device}")

        # ========================================================
        # CHECK FILES
        # ========================================================

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found:\n{checkpoint_path}"
            )

        if not os.path.exists(acoustic_stats_path):
            raise FileNotFoundError(
                f"Acoustic stats file not found:\n{acoustic_stats_path}"
            )

        # ========================================================
        # CREATE MODEL
        # ========================================================

        self.model = MultiModalAcousticNet().to(self.device)

        # ========================================================
        # LOAD CHECKPOINT
        # ========================================================

        print("[AcousticSpace] Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # ========================================================
        # HANDLE DATA PARALLEL
        # ========================================================

        if any(
            isinstance(k, str) and k.startswith("module.")
            for k in state_dict.keys()
        ):
            print(
                "[AcousticSpace] DataParallel checkpoint detected."
            )

            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }

            print(
                "[AcousticSpace] Removed 'module.' prefix."
            )

        # ========================================================
        # REPAIR CORRUPTED BATCHNORM BUFFERS
        # ========================================================
        #
        # Your existing checkpoint contains non-finite values in:
        #
        # acoustic_branch.mlp.1.running_mean
        # acoustic_branch.mlp.1.running_var
        # acoustic_branch.mlp.4.running_mean
        # acoustic_branch.mlp.4.running_var
        # classifier.classifier.1.running_mean
        # classifier.classifier.1.running_var
        # classifier.classifier.5.running_mean
        # classifier.classifier.5.running_var
        #
        # These buffers cause:
        #       logits = [nan, nan]
        #
        # We repair ONLY BatchNorm running statistics.
        # We do NOT modify learned weights/biases.
        # ========================================================

        repaired_buffers = []

        for key, value in list(state_dict.items()):

            if not torch.is_tensor(value):
                continue

            if not torch.isfinite(value).all():

                if key.endswith("running_mean"):
                    fixed = torch.nan_to_num(
                        value,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )

                    state_dict[key] = fixed

                    repaired_buffers.append(
                        f"{key} -> non-finite values replaced with 0"
                    )

                elif key.endswith("running_var"):
                    fixed = torch.nan_to_num(
                        value,
                        nan=1.0,
                        posinf=1.0,
                        neginf=1.0
                    )

                    # BatchNorm variance must be positive.
                    fixed = torch.clamp(
                        fixed,
                        min=1e-5
                    )

                    state_dict[key] = fixed

                    repaired_buffers.append(
                        f"{key} -> non-finite values replaced with 1"
                    )

                else:
                    raise RuntimeError(
                        "Checkpoint contains non-finite learned "
                        f"parameter/buffer that cannot safely be repaired: "
                        f"{key}"
                    )

        if repaired_buffers:
            print(
                "[AcousticSpace] WARNING: repaired corrupted "
                "BatchNorm buffers in memory:"
            )

            for item in repaired_buffers:
                print("   ", item)

        # ========================================================
        # FINAL CHECKPOINT VALIDATION
        # ========================================================

        bad_after_repair = []

        for key, value in state_dict.items():
            if (
                torch.is_tensor(value)
                and not torch.isfinite(value).all()
            ):
                bad_after_repair.append(key)

        if bad_after_repair:
            raise RuntimeError(
                "Checkpoint still contains non-finite tensors after "
                f"repair: {bad_after_repair}"
            )

        # ========================================================
        # LOAD WEIGHTS
        # ========================================================

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.eval()

        print(
            "[AcousticSpace] Model loaded successfully."
        )

        # ========================================================
        # LOAD ACOUSTIC NORMALIZATION
        # ========================================================

        with open(
            acoustic_stats_path,
            "r",
            encoding="utf-8"
        ) as f:
            stats = json.load(f)

        self.acoustic_mean = np.asarray(
            stats["acoustic_mean"],
            dtype=np.float32
        )

        self.acoustic_std = np.asarray(
            stats["acoustic_std"],
            dtype=np.float32
        )

        if self.acoustic_mean.shape[0] != 9:
            raise ValueError(
                "acoustic_mean must contain exactly 9 values."
            )

        if self.acoustic_std.shape[0] != 9:
            raise ValueError(
                "acoustic_std must contain exactly 9 values."
            )

        # ========================================================
        # REPAIR INVALID ACOUSTIC STATS
        # ========================================================
        #
        # Existing acoustic_stats.json contains:
        #
        # mean[8] = NaN
        # std[8]  = NaN
        #
        # Feature 8 is breath_variation/std_interval.
        #
        # For this emergency deployment:
        #
        #   mean = 0
        #   std  = 1
        #
        # This means the 9th feature is passed through without
        # applying an invalid training normalization.
        # ========================================================

        invalid_mean = ~np.isfinite(self.acoustic_mean)
        invalid_std = ~np.isfinite(self.acoustic_std)

        if invalid_mean.any():
            print(
                "[AcousticSpace] WARNING: invalid acoustic_mean "
                "detected."
            )

            for idx in np.where(invalid_mean)[0]:
                print(
                    f"   acoustic_mean[{idx}] -> 0.0"
                )

            self.acoustic_mean[invalid_mean] = 0.0

        if invalid_std.any():
            print(
                "[AcousticSpace] WARNING: invalid acoustic_std "
                "detected."
            )

            for idx in np.where(invalid_std)[0]:
                print(
                    f"   acoustic_std[{idx}] -> 1.0"
                )

            self.acoustic_std[invalid_std] = 1.0

        # Prevent zero / negative standard deviations.
        self.acoustic_std = np.where(
            self.acoustic_std > 1e-8,
            self.acoustic_std,
            1.0
        ).astype(np.float32)

        print(
            "[AcousticSpace] Acoustic normalization statistics "
            "loaded successfully."
        )

        print(
            "[AcousticSpace] Acoustic mean:",
            self.acoustic_mean
        )

        print(
            "[AcousticSpace] Acoustic std:",
            self.acoustic_std
        )

    # ============================================================
    # NORMALIZATION HELPERS
    # ============================================================

    @staticmethod
    def normalize_per_sample(
        array: np.ndarray
    ) -> np.ndarray:
        """
        Per-sample normalization for:
            - spectrogram
            - room descriptor
        """

        array = np.asarray(
            array,
            dtype=np.float32
        )

        # Repair any unexpected preprocessing NaN/Inf BEFORE
        # calculating mean/std.
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        mean = float(np.mean(array))
        std = float(np.std(array))

        normalized = (
            array - mean
        ) / (
            std + 1e-8
        )

        return np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

    def normalize_acoustic(
        self,
        acoustic: np.ndarray
    ) -> np.ndarray:
        """
        Normalize the 9 acoustic features.

        Feature order from preprocess.py:
            0 room_energy
            1 room_clarity_c50
            2 room_peak_index
            3 room_centroid
            4 drr_db
            5 rt60_sec
            6 breath_count
            7 avg_breath_interval
            8 breath_variation
        """

        acoustic = np.asarray(
            acoustic,
            dtype=np.float32
        )

        if acoustic.shape[0] != 9:
            raise ValueError(
                "Expected 9 acoustic features, "
                f"received {acoustic.shape[0]}."
            )

        # Repair unexpected input feature NaN/Inf.
        acoustic = np.nan_to_num(
            acoustic,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        normalized = (
            acoustic - self.acoustic_mean
        ) / (
            self.acoustic_std + 1e-8
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return normalized.astype(np.float32)

    # ============================================================
    # SAFE FEATURE AVERAGING
    # ============================================================

    @staticmethod
    def average_feature(
        segments,
        key: str
    ) -> float:

        values = np.asarray(
            [
                segment.get(key, 0.0)
                for segment in segments
            ],
            dtype=np.float32
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return float(np.mean(values))

    # ============================================================
    # MAIN INFERENCE
    # ============================================================

    @torch.no_grad()
    def analyze_file(
        self,
        audio_path: str,
        original_filename: str
    ) -> dict:

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        work_dir = tempfile.mkdtemp(
            prefix="acousticspace_infer_"
        )

        try:

            # ====================================================
            # 1. PREPROCESS AUDIO
            # ====================================================

            print("\n" + "=" * 70)
            print("ACOUSTICSPACE ANALYSIS")
            print("=" * 70)

            print(
                "Original filename :",
                original_filename
            )

            print(
                "Temporary file   :",
                audio_path
            )

            print(
                "Device            :",
                self.device
            )

            print("-" * 70)

            json_path = audio_to_spectrogram(
                audio_path,
                output_dir=work_dir
            )

            if not os.path.exists(json_path):
                raise RuntimeError(
                    "Preprocessing did not generate "
                    "the expected analysis JSON."
                )

            # ====================================================
            # 2. LOAD REPORT
            # ====================================================

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:
                report = json.load(f)

            segments = report.get(
                "segments",
                []
            )

            if len(segments) == 0:
                raise ValueError(
                    "No speech segments detected in this file. "
                    "The audio may be silent, too quiet, or "
                    "the VAD could not detect speech."
                )

            # ====================================================
            # 3. PREPARE BATCHES
            # ====================================================

            spec_batch = []
            room_batch = []
            acoustic_batch = []

            for segment in segments:

                # ------------------------------------------------
                # Spectrogram
                # ------------------------------------------------

                spec = np.load(
                    segment["spectrogram_path"]
                ).astype(np.float32)

                if spec.ndim != 2:
                    raise ValueError(
                        "Expected spectrogram with shape "
                        "(128, T), got "
                        f"{spec.shape}"
                    )

                spec = self.normalize_per_sample(
                    spec
                )

                # ------------------------------------------------
                # Room descriptor
                # ------------------------------------------------

                room = np.load(
                    segment["room_descriptor_path"]
                ).astype(np.float32)

                room = self.normalize_per_sample(
                    room
                )

                # ------------------------------------------------
                # Acoustic features
                # ------------------------------------------------

                acoustic = np.asarray(
                    segment["acoustic_features"],
                    dtype=np.float32
                )

                acoustic = self.normalize_acoustic(
                    acoustic
                )

                # ------------------------------------------------
                # Append
                # ------------------------------------------------

                spec_batch.append(
                    spec[np.newaxis, :, :]
                )

                room_batch.append(
                    room
                )

                acoustic_batch.append(
                    acoustic
                )

            # ====================================================
            # 4. PAD SPECTROGRAMS
            # ====================================================

            max_t = max(
                spec.shape[-1]
                for spec in spec_batch
            )

            padded_spec_batch = []

            for spec in spec_batch:

                current_t = spec.shape[-1]

                if current_t < max_t:

                    spec = np.pad(
                        spec,
                        (
                            (0, 0),
                            (0, 0),
                            (0, max_t - current_t)
                        ),
                        mode="constant"
                    )

                padded_spec_batch.append(
                    spec
                )

            # ====================================================
            # 5. TORCH TENSORS
            # ====================================================

            spec_tensor = torch.from_numpy(
                np.stack(padded_spec_batch)
            ).float().to(self.device)

            room_tensor = torch.from_numpy(
                np.stack(room_batch)
            ).float().to(self.device)

            acoustic_tensor = torch.from_numpy(
                np.stack(acoustic_batch)
            ).float().to(self.device)

            # ====================================================
            # 6. INPUT VALIDATION
            # ====================================================

            print("\n========== INPUT VALIDATION ==========")

            print(
                "Spec shape:",
                spec_tensor.shape
            )

            print(
                "Room shape:",
                room_tensor.shape
            )

            print(
                "Acoustic shape:",
                acoustic_tensor.shape
            )

            spec_finite = bool(
                torch.isfinite(spec_tensor).all().item()
            )

            room_finite = bool(
                torch.isfinite(room_tensor).all().item()
            )

            acoustic_finite = bool(
                torch.isfinite(acoustic_tensor).all().item()
            )

            print(
                "Spec finite:",
                spec_finite
            )

            print(
                "Room finite:",
                room_finite
            )

            print(
                "Acoustic finite:",
                acoustic_finite
            )

            if not spec_finite:
                raise RuntimeError(
                    "Spectrogram tensor contains NaN or Inf."
                )

            if not room_finite:
                raise RuntimeError(
                    "Room tensor contains NaN or Inf."
                )

            if not acoustic_finite:
                raise RuntimeError(
                    "Acoustic tensor contains NaN or Inf."
                )

            print(
                "=======================================\n"
            )

            # ====================================================
            # 7. MODEL INFERENCE
            # ====================================================

            logits = self.model(
                spec_tensor,
                room_tensor,
                acoustic_tensor
            )

            print(
                "========== MODEL OUTPUT =========="
            )

            print(
                "Logits:",
                logits
            )

            logits_finite = bool(
                torch.isfinite(logits).all().item()
            )

            print(
                "Logits finite:",
                logits_finite
            )

            if not logits_finite:
                raise RuntimeError(
                    "Model produced invalid logits after "
                    "checkpoint repair: "
                    f"{logits.detach().cpu().numpy()}"
                )

            # ====================================================
            # 8. SOFTMAX
            # ====================================================

            probs = F.softmax(
                logits,
                dim=1
            )

            probs = torch.nan_to_num(
                probs,
                nan=0.5,
                posinf=1.0,
                neginf=0.0
            )

            # Re-normalize after emergency safety conversion.
            probs_sum = probs.sum(
                dim=1,
                keepdim=True
            ).clamp_min(1e-8)

            probs = probs / probs_sum

            print(
                "Probabilities:",
                probs
            )

            if not torch.isfinite(probs).all():
                raise RuntimeError(
                    "Model produced invalid probabilities."
                )

            # ====================================================
            # 9. SPOOF PROBABILITY
            # ====================================================
            #
            # Dataset label mapping:
            #   0 = authentic / bonafide
            #   1 = synthetic / spoof
            # ====================================================

            spoof_probs = (
                probs[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if not np.isfinite(spoof_probs).all():
                raise RuntimeError(
                    "Invalid spoof probabilities."
                )

            spoof_probs = np.clip(
                spoof_probs,
                0.0,
                1.0
            )

            # ====================================================
            # 10. FILE-LEVEL RESULT
            # ====================================================

            mean_spoof_prob = float(
                np.mean(spoof_probs)
            )

            mean_spoof_prob = float(
                np.clip(
                    mean_spoof_prob,
                    0.0,
                    1.0
                )
            )

            if mean_spoof_prob >= 0.5:
                verdict = "synthetic"
                confidence_pct = (
                    mean_spoof_prob * 100.0
                )
            else:
                verdict = "authentic"
                confidence_pct = (
                    (1.0 - mean_spoof_prob)
                    * 100.0
                )

            confidence_pct = float(
                np.clip(
                    confidence_pct,
                    0.0,
                    100.0
                )
            )

            print(
                "Spoof probabilities:",
                spoof_probs
            )

            print(
                "Mean spoof probability:",
                mean_spoof_prob
            )

            print(
                "Verdict:",
                verdict
            )

            print(
                "Confidence:",
                confidence_pct
            )

            print(
                "=================================\n"
            )

            # ====================================================
            # 11. ACOUSTIC SUMMARY
            # ====================================================

            acoustic_summary = {
                "rt60_sec": self.average_feature(
                    segments,
                    "rt60_sec"
                ),

                "drr_db": self.average_feature(
                    segments,
                    "drr_db"
                ),

                "room_clarity_c50": self.average_feature(
                    segments,
                    "room_clarity_c50"
                ),

                "room_centroid": self.average_feature(
                    segments,
                    "room_centroid"
                ),

                "breath_count": self.average_feature(
                    segments,
                    "breath_count"
                ),

                "avg_breath_interval": self.average_feature(
                    segments,
                    "avg_breath_interval"
                ),

                # Extra internal features are returned too.
                "room_energy": self.average_feature(
                    segments,
                    "room_energy"
                ),

                "room_peak_index": self.average_feature(
                    segments,
                    "room_peak_index"
                ),

                "breath_variation": self.average_feature(
                    segments,
                    "breath_variation"
                ),
            }

            # ====================================================
            # 12. SEGMENT PREDICTIONS
            # ====================================================

            segment_predictions = []

            for segment, probability in zip(
                segments,
                spoof_probs
            ):

                probability = float(
                    np.clip(
                        probability,
                        0.0,
                        1.0
                    )
                )

                if probability >= 0.5:
                    segment_verdict = "synthetic"
                else:
                    segment_verdict = "authentic"

                segment_confidence = float(
                    max(
                        probability,
                        1.0 - probability
                    ) * 100.0
                )

                segment_predictions.append({
                    "segment_id": int(
                        segment["segment_id"]
                    ),
                    "spoof_probability": probability,
                    "verdict": segment_verdict,
                    "confidence_pct": segment_confidence,
                })

            # ====================================================
            # 13. AUDIO METADATA
            # ====================================================

            duration_sec = float(
                librosa.get_duration(
                    path=audio_path
                )
            )

            sample_rate = int(
                report.get(
                    "metadata",
                    {}
                ).get(
                    "sample_rate",
                    16000
                )
            )

            is_clipped = bool(
                report.get(
                    "quality",
                    {}
                ).get(
                    "is_clipped",
                    False
                )
            )

            # ====================================================
            # 14. FINAL RESPONSE
            # ====================================================

            result = {
                "filename": original_filename,

                "duration_sec": duration_sec,

                "sample_rate": sample_rate,

                "num_segments": len(segments),

                "is_clipped": is_clipped,

                "verdict": verdict,

                "confidence_pct": float(
                    confidence_pct
                ),

                "spoof_probability": float(
                    mean_spoof_prob
                ),

                "acoustic_summary": acoustic_summary,

                "segment_predictions": (
                    segment_predictions
                ),
            }

            return result

        finally:

            shutil.rmtree(
                work_dir,
                ignore_errors=True
            )