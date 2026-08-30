from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class EqBand:
    frequency: int
    q: float
    gain_db: float
    confidence: float


# Broad bands only. The profile values are filled from the clean male U87 Ai
# reference and normalized over the speech body, so no reference audio needs
# to be distributed with the application.
BANDS = (
    (120, 0.65),
    (180, 0.60),
    (380, 0.55),
    (900, 0.50),
    (2500, 0.55),
    (4200, 0.60),
    (6800, 0.55),
    (9500, 0.50),
)

# Replaced below after measuring the selected U87 reference. Keeping this as
# data rather than audio makes the matching reproducible and redistributable.
U87_PROFILE_DB = np.array(
    [9.254663, 10.244543, 8.396354, 3.652617, -3.652617, -9.340652, -12.042131, -13.094386],
    dtype=np.float64,
)


def _speech_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    audio, sample_rate = librosa.load(path, sr=48000, mono=True)
    spectrum = np.abs(
        librosa.stft(audio, n_fft=4096, hop_length=480, win_length=4096, window="hann")
    ) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=4096)

    speech = spectrum[(frequencies >= 100) & (frequencies < 8000)].sum(axis=0)
    db = 10.0 * np.log10(speech + 1e-20)
    flatness = librosa.feature.spectral_flatness(S=spectrum).ravel()
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(spectrum), sr=sample_rate).ravel()
    active = (db > np.percentile(db, 45)) & (flatness < 0.24) & (centroid < 4200)
    if active.sum() < 12:
        active = db > np.percentile(db, 55)
    quiet = db < np.percentile(db, 20)

    levels: list[float] = []
    confidence: list[float] = []
    for center, q in BANDS:
        # A smooth log-frequency weighting avoids matching individual formants
        # or FFT peaks. q here controls an intentionally broad analysis band.
        width_octaves = 1.0 / q
        distance = np.log2(np.maximum(frequencies, 1.0) / center)
        weights = np.exp(-0.5 * (distance / (width_octaves / 2.355)) ** 2)
        weights[(frequencies < 70) | (frequencies > 12000)] = 0
        band_power = (spectrum * weights[:, None]).sum(axis=0)
        active_level = float(np.median(10.0 * np.log10(band_power[active] + 1e-20)))
        quiet_level = float(np.median(10.0 * np.log10(band_power[quiet] + 1e-20)))
        levels.append(active_level)
        confidence.append(float(np.clip((active_level - quiet_level - 6.0) / 24.0, 0.0, 1.0)))

    levels_array = np.asarray(levels)
    # Remove recording level using the robust center of the voice-body bands.
    levels_array -= np.median(levels_array[2:6])
    return levels_array, np.asarray(confidence)


def reference_profile(path: Path) -> np.ndarray:
    return _speech_profile(path)[0]


def adaptive_u87_bands(source: Path, reference: Path | None = None) -> list[EqBand]:
    source_profile, confidence = _speech_profile(source)
    target = reference_profile(reference) if reference is not None else U87_PROFILE_DB
    correction = target - source_profile

    # Smooth adjacent bands to reject speaker/formant differences. A single
    # pass is intentional: iterative matching tends to overfit phonemes.
    padded = np.pad(correction, (1, 1), mode="edge")
    correction = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    correction = np.clip(correction, -3.0, 3.0)

    result: list[EqBand] = []
    for index, ((frequency, q), gain, certainty) in enumerate(zip(BANDS, correction, confidence)):
        # Do not synthesize missing air. Positive high-band corrections are
        # limited much more strongly than cuts and scale with measured SNR.
        positive_limit = 3.0
        if frequency >= 4200:
            positive_limit = 0.0
        gain = min(float(gain), positive_limit * float(certainty))
        gain = max(gain, -3.0)
        if abs(gain) < 0.25:
            gain = 0.0
        result.append(EqBand(frequency, q, round(gain, 2), round(float(certainty), 2)))

    # Broad parametric bands overlap. Limit their combined positive and
    # negative budgets so individually safe bands cannot sum into a large
    # low shelf or presence scoop.
    positive_total = sum(max(band.gain_db, 0.0) for band in result)
    negative_total = sum(max(-band.gain_db, 0.0) for band in result)
    positive_scale = min(1.0, 3.0 / positive_total) if positive_total else 1.0
    negative_scale = min(1.0, 3.0 / negative_total) if negative_total else 1.0
    limited: list[EqBand] = []
    for band in result:
        scale = positive_scale if band.gain_db >= 0 else negative_scale
        gain = round(band.gain_db * scale, 2)
        if abs(gain) < 0.2:
            gain = 0.0
        limited.append(EqBand(band.frequency, band.q, gain, band.confidence))
    return limited


def sibilance_center(path: Path) -> int:
    """Locate the dominant narrow S energy without treating broadband air as S."""
    audio, sample_rate = librosa.load(path, sr=48000, mono=True)
    spectrum = np.abs(librosa.stft(audio, n_fft=4096, hop_length=240, window="hann")) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=4096)
    voice = spectrum[(frequencies >= 150) & (frequencies < 4500)].sum(axis=0)
    upper = spectrum[(frequencies >= 4500) & (frequencies < 12000)].sum(axis=0)
    ratio = 10.0 * np.log10((upper + 1e-20) / (voice + 1e-20))
    energy = 10.0 * np.log10(upper + voice + 1e-20)
    selected = (ratio >= np.percentile(ratio, 90)) & (energy >= np.percentile(energy, 45))
    band = (frequencies >= 4800) & (frequencies <= 10500)
    profile = np.median(spectrum[band][:, selected], axis=1)
    center = int(round(float(frequencies[band][int(np.argmax(profile))]) / 50.0) * 50)
    return max(4800, min(10500, center))


def ffmpeg_filters(bands: list[EqBand]) -> list[str]:
    return [
        f"equalizer=f={band.frequency}:t=q:w={band.q:g}:g={band.gain_db:g}"
        for band in bands
        if band.gain_db != 0
    ]


def describe_bands(bands: list[EqBand]) -> list[str]:
    return [
        f"{band.frequency / 1000:g} kHz: {band.gain_db:+g} dB "
        f"({band.confidence * 100:.0f}% confidence)"
        for band in bands
        if band.gain_db != 0
    ]
