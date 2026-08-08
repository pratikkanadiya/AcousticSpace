import os
import json
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly


def _levinson_durbin(r: torch.Tensor, order: int):

    device = r.device
    dtype = r.dtype

    a = torch.zeros(order + 1, device=device, dtype=dtype)
    a[0] = 1.0

    err = r[0]

    if err <= 0:
        return a, torch.tensor(0.0, device=device)

    for i in range(1, order + 1):

        if i == 1:
            acc = r[i]
        else:
            acc = r[i] + torch.dot(
                a[1:i],
                torch.flip(r[1:i], dims=[0])
            )

        k = -acc / (err + 1e-12)

        a_prev = a.clone()

        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]

        a[i] = k

        err = err * (1.0 - k * k)

        if err <= 1e-12:
            break

    return a, err


def compute_lpc_residual(
    segment: torch.Tensor,
    sr: int,
    order: int = None
):

    device = segment.device

    segment = segment.float()

    if order is None:
        order = sr // 1000 + 2

    order = max(2, min(order, segment.numel() - 1))

    if torch.max(torch.abs(segment)) < 1e-8:
        return segment

    x = segment.view(1, 1, -1)

    corr = F.conv1d(
        x,
        torch.flip(x, dims=[2]),
        padding=segment.numel() - 1
    )

    corr = corr.view(-1)

    mid = corr.numel() // 2

    r = corr[mid:mid + order + 1]

    if r[0] <= 0:
        return segment


    a, err = _levinson_durbin(r, order)


    kernel = a.view(1, 1, -1)

    residual = F.conv1d(
        x,
        kernel,
        padding=order
    )

    residual = residual.view(-1)[:segment.numel()]

    residual = torch.nan_to_num(
        residual,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return residual


import torch
import torch.nn.functional as F


def estimate_room_response_descriptor(
    audio,
    sr,
    lpc_order=None,
    smoothing_window=15
):
    """
    GPU version of room response estimation.

    audio : torch.Tensor (CUDA or CPU)
    """

    residual = compute_lpc_residual(audio, sr, order=lpc_order)


    x = residual.unsqueeze(0).unsqueeze(0)

    corr = F.conv1d(
        x,
        torch.flip(x, dims=[2]),
        padding=residual.numel() - 1
    ).squeeze()

    corr = corr[corr.numel() // 2:]

    corr = corr / (torch.max(torch.abs(corr)) + 1e-8)


    window = min(smoothing_window, corr.numel())

    if window % 2 == 0:
        window -= 1

    if window >= 5:

        corr = F.avg_pool1d(
            corr.unsqueeze(0).unsqueeze(0),
            kernel_size=window,
            stride=1,
            padding=window // 2
        ).squeeze()

    room_descriptor = corr


    max_samples = int(0.5 * sr)

    room_descriptor = room_descriptor[:max_samples]


    if room_descriptor.numel() < max_samples:

        pad = max_samples - room_descriptor.numel()

        room_descriptor = F.pad(
            room_descriptor,
            (0, pad)
        )

    return room_descriptor


def extract_room_response_features(room, sr):
    """
    GPU version
    """

    energy = torch.sum(room ** 2)

    peak = torch.argmax(torch.abs(room))

    split = int(0.05 * sr)

    early = room[:split]

    late = room[split:]

    early_energy = torch.sum(early ** 2)

    late_energy = torch.sum(late ** 2)

    clarity = 10 * torch.log10(
        (early_energy + 1e-8) /
        (late_energy + 1e-8)
    )

    idx = torch.arange(
        room.numel(),
        device=room.device,
        dtype=room.dtype
    )

    centroid = (
        torch.sum(idx * torch.abs(room))
        /
        (torch.sum(torch.abs(room)) + 1e-8)
    )

    return {

        "room_energy": energy.item(),

        "room_clarity_c50": clarity.item(),

        "room_peak_index": int(peak.item()),

        "room_centroid": centroid.item()

    }


def bandpass(signal_data, sr):

    device = signal_data.device

    filtered = torchaudio.functional.bandpass_biquad(
        signal_data.unsqueeze(0),
        sample_rate=sr,
        central_freq=550.0,
        Q=0.61
    )

    return filtered.squeeze(0)


def rms_feature(audio, frame, hop):

    frames = audio.unfold(0, frame, hop)

    rms = torch.sqrt(
        torch.mean(frames ** 2, dim=1) + 1e-8
    )

    return rms


def spectral_centroid(audio, sr, frame, hop):

    window = torch.hann_window(
        frame,
        device=audio.device
    )

    spec = torch.stft(
        audio,
        n_fft=frame,
        hop_length=hop,
        win_length=frame,
        window=window,
        return_complex=True
    )

    mag = spec.abs()

    freqs = torch.linspace(
        0,
        sr / 2,
        mag.shape[0],
        device=audio.device
    )

    centroid = (
        freqs[:, None] * mag
    ).sum(0)

    centroid /= (
        mag.sum(0) + 1e-8
    )

    return centroid


def zero_crossing_rate(audio, frame, hop):

    frames = audio.unfold(0, frame, hop)

    signs = torch.sign(frames)

    zcr = (
        signs[:, :-1] != signs[:, 1:]
    ).float().mean(dim=1)

    return zcr


def detect_breathing(filtered_audio,
                     original_audio,
                     sr):

    frame = int(0.03 * sr)

    hop = int(0.015 * sr)

    rms = rms_feature(
        filtered_audio,
        frame,
        hop
    )

    centroid = spectral_centroid(
        original_audio,
        sr,
        frame,
        hop
    )

    zcr = zero_crossing_rate(
        filtered_audio,
        frame,
        hop
    )

    n = min(
        rms.numel(),
        centroid.numel(),
        zcr.numel()
    )

    rms = rms[:n]

    centroid = centroid[:n]

    zcr = zcr[:n]

    rms_threshold = torch.quantile(
        rms,
        0.35
    )

    mask = (
        (rms < rms_threshold)
        &
        (centroid < 1800)
        &
        (zcr > 0.08)
    )

    breath_frames = torch.nonzero(
        mask
    ).flatten()

    breath_times = (
        breath_frames.float() * hop
    ) / sr

    return breath_times


def breathing_features(breath_times):

    if breath_times.numel() < 2:

        return {

            "breath_count": int(breath_times.numel()),

            "avg_interval": 0.0,

            "std_interval": 0.0

        }

    intervals = breath_times[1:] - breath_times[:-1]

    return {

        "breath_count": int(breath_times.numel()),

        "avg_interval": intervals.mean().item(),

        "std_interval": intervals.std().item()

    }


def detect_clipping(audio_tensor, threshold=0.99):
    """
    PyTorch version of clipping detection.

    Args:
        audio_tensor : torch.Tensor
        threshold    : clipping threshold

    Returns:
        bool
    """

    return bool(torch.max(torch.abs(audio_tensor)) >= threshold)


def estimate_drr(segment, sr, direct_window_ms=2.5):
    """
    Pure PyTorch implementation.
    """

    if segment.numel() == 0:
        return 0.0

    peak_idx = torch.argmax(torch.abs(segment)).item()

    window = int((direct_window_ms / 1000.0) * sr)

    start = max(0, peak_idx - window)
    end = min(segment.numel(), peak_idx + window)

    direct_energy = torch.sum(segment[start:end] ** 2)

    reverberant_energy = (
        torch.sum(segment[:start] ** 2)
        +
        torch.sum(segment[end:] ** 2)
    )

    drr = 10 * torch.log10(
        (direct_energy + 1e-12)
        /
        (reverberant_energy + 1e-12)
    )

    return round(drr.item(), 2)


def estimate_rt60(segment, sr):
    """
    Pure PyTorch implementation.
    """

    if segment.numel() < int(0.1 * sr):
        return 0.0

    energy = segment ** 2

    edc = torch.cumsum(
        torch.flip(energy, dims=[0]),
        dim=0
    )

    edc = torch.flip(edc, dims=[0])

    edc = edc / (torch.max(edc) + 1e-12)

    edc_db = 10 * torch.log10(edc + 1e-12)

    idx = torch.where(
        (edc_db <= -5)
        &
        (edc_db >= -35)
    )[0]

    if idx.numel() < 20:
        return 0.0

    x = idx.float() / sr

    y = edc_db[idx]


    A = torch.stack(
        [x, torch.ones_like(x)],
        dim=1
    )

    solution = torch.linalg.lstsq(A, y.unsqueeze(1)).solution

    slope = solution[0, 0]

    if slope >= 0:
        return 0.0

    t30 = -30.0 / slope

    rt60 = t30 * 2.0

    return round(rt60.item(), 3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def audio_to_spectrogram(
    audio_path,
    output_dir="acoustic_reports",
    sr=16000,
    seg_len_sec=5.0
):

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    report = {

        "metadata": {

            "filename": os.path.basename(audio_path),

            "sample_rate": sr

        },

        "quality": {},

        "segments": []

    }


    import soundfile as sf
    from scipy.signal import resample_poly

    audio_np, sample_rate = sf.read(audio_path, dtype="float32")

    # Convert stereo -> mono
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    # Resample
    if sample_rate != sr:
        audio_np = resample_poly(
            audio_np,
            sr,
            sample_rate
        ).astype(np.float32)

    audio = torch.from_numpy(audio_np).to(device)


    report["quality"]["is_clipped"] = detect_clipping(audio)

    audio = audio / (audio.abs().max() + 1e-8)

    report["quality"]["normalized"] = True


    intervals = torchaudio.functional.vad(
        audio.unsqueeze(0),
        sample_rate=sr
    )

    if intervals.numel() == 0:

        vad_intervals = [(0, audio.numel())]

    else:

        vad_intervals = [(0, intervals.shape[-1])]


    samples_per_seg = int(seg_len_sec * sr)

    segment_count = 0

    mel_transform = torchaudio.transforms.MelSpectrogram(

        sample_rate=sr,

        n_fft=1024,

        win_length=1024,

        hop_length=256,

        n_mels=128,

        f_min=20,

        f_max=8000

    ).to(device)

    db_transform = torchaudio.transforms.AmplitudeToDB().to(device)

    for start, end in vad_intervals:

        speech_chunk = audio[start:end]

        for i in range(
            0,
            speech_chunk.numel(),
            samples_per_seg
        ):

            segment = speech_chunk[
                i:i + samples_per_seg
            ]


            if segment.numel() < samples_per_seg:

                segment = F.pad(

                    segment,

                    (0, samples_per_seg - segment.numel())

                )


            room = estimate_room_response_descriptor(
                segment,
                sr
            )

            room_features = extract_room_response_features(
                room,
                sr
            )


            filtered_segment = bandpass(
                segment,
                sr
            )

            breaths = detect_breathing(

                filtered_segment,

                segment,

                sr

            )

            breath_stats = breathing_features(
                breaths
            )


            drr = estimate_drr(segment, sr)

            rt60 = estimate_rt60(segment, sr)

            mel_spec = mel_transform(
                segment.unsqueeze(0)
            )

            log_mel_spec = db_transform(
                mel_spec
            )

            log_mel_spec = log_mel_spec.squeeze(0)

            segment_count += 1

            acoustic_features = [

                room_features["room_energy"],

                room_features["room_clarity_c50"],

                room_features["room_peak_index"],

                room_features["room_centroid"],

                drr,

                rt60,

                breath_stats["breath_count"],

                breath_stats["avg_interval"],

                breath_stats["std_interval"]

            ]


            acoustic_features = np.asarray(
                acoustic_features,
                dtype=np.float32
            )

            if not np.isfinite(acoustic_features).all():
                print(
                    f"[WARNING] Invalid acoustic features detected "
                    f"for {base_name}, segment {segment_count}"
                )

                print("Original acoustic features:")
                print(acoustic_features)

                # Replace invalid values safely
                acoustic_features = np.nan_to_num(
                    acoustic_features,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

            acoustic_features = acoustic_features.tolist()
            

            spec_path = os.path.join(

                output_dir,

                f"{base_name}_seg{segment_count}_spec.npy"

            )

            np.save(

                spec_path,

                log_mel_spec.cpu().numpy().astype(np.float32)

            )

            room_descriptor_path = os.path.join(

                output_dir,

                f"{base_name}_seg{segment_count}_room.npy"

            )

            np.save(

                room_descriptor_path,

                room.cpu().numpy().astype(np.float32)

            )

            report["segments"].append({

                "segment_id": segment_count,

                "drr_db": drr,

                "rt60_sec": rt60,

                "room_energy": room_features["room_energy"],

                "room_clarity_c50": room_features["room_clarity_c50"],

                "room_peak_index": room_features["room_peak_index"],

                "room_centroid": room_features["room_centroid"],

                "breath_count": breath_stats["breath_count"],

                "avg_breath_interval": breath_stats["avg_interval"],

                "breath_variation": breath_stats["std_interval"],

                "spectrogram_path": spec_path,

                "room_descriptor_path": room_descriptor_path,

                "acoustic_features": acoustic_features

            })

    report["total_segments"] = segment_count

    json_path = os.path.join(

        output_dir,

        f"{base_name}_analysis.json"

    )

    with open(json_path, "w") as f:

        json.dump(

            report,

            f,

            indent=4

        )

    return json_path