"""80 Hz high-pass, ButterComp2, transparent EQ, and DeBess in one pass."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from .buttercomp_runner import ButterComp2


def peaking_sos(sample_rate: int, frequency: float, q: float, gain_db: float) -> np.ndarray:
    """Robert Bristow-Johnson peaking EQ biquad, normalized as scipy SOS."""
    amplitude = math.pow(10.0, gain_db / 40.0)
    omega = 2.0 * math.pi * frequency / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    cosine = math.cos(omega)
    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cosine
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha / amplitude
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def transparent_eq(audio: np.ndarray, sample_rate: int, bands: list[dict[str, float]]) -> np.ndarray:
    sections = list(
        peaking_sos(sample_rate, band["frequency"], band["q"], band["gain_db"])
        for band in bands
        if band["gain_db"] != 0
    )
    if not sections:
        return audio.astype(np.float64, copy=True)
    return sosfilt(np.vstack(sections), audio).astype(np.float64, copy=False)


def highpass_80hz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    sections = butter(2, 80.0, btype="highpass", fs=sample_rate, output="sos")
    return sosfilt(sections, audio).astype(np.float64, copy=False)


class DeBess:
    """Mono double-precision adaptation of Airwindows DeBess."""

    def __init__(
        self, sample_rate: int, center_hz: float, intensity: float = 0.35,
        depth_control: float = 0.67,
    ):
        overall_scale = sample_rate / 44100.0
        self.intensity = math.pow(intensity, 5) * (8192.0 / overall_scale)
        self.sharpness = 20
        self.speed = 0.1 / self.sharpness
        self.depth = 1.0 / (depth_control + 0.0001)
        self.iir_amount = 1.0 - math.exp(-2.0 * math.pi * center_hz / sample_rate)
        self.samples = np.zeros(42, dtype=np.float64)
        self.slews = np.zeros(42, dtype=np.float64)
        self.ratio_a = self.ratio_b = 1.0
        self.iir_a = self.iir_b = 0.0
        self.flip = False

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        output = np.empty_like(audio, dtype=np.float64)
        removed = np.empty_like(audio, dtype=np.float64)
        ratios = np.ones(len(audio), dtype=np.float64)
        for index, original in enumerate(audio.astype(np.float64, copy=False)):
            self.samples[1 : self.sharpness + 1] = self.samples[: self.sharpness]
            self.samples[0] = original
            self.slews[1] = (self.samples[1] - self.samples[2]) ** 2 / 1.3
            for position in range(self.sharpness - 1, 1, -1):
                self.slews[position] = (
                    (self.samples[position] - self.samples[position + 1])
                    * (self.samples[position - 1] - self.samples[position])
                    / 1.3
                )
            sense = abs(self.slews[1] - self.slews[2]) * self.sharpness**2
            for position in range(self.sharpness - 1, 0, -1):
                multiplier = abs(self.slews[position] - self.slews[position + 1]) * self.sharpness**2
                if multiplier < 1.0:
                    sense *= multiplier
            sense = min(1.0 + self.intensity * self.intensity * sense, self.intensity)

            if self.flip:
                self.iir_a = self.iir_a * (1.0 - self.iir_amount) + original * self.iir_amount
                self.ratio_a = self.ratio_a * (1.0 - self.speed) + sense * self.speed
                self.ratio_a = min(self.ratio_a, self.depth)
                ratio = self.ratio_a
                processed = self.iir_a + (original - self.iir_a) / ratio if ratio > 1.0 else original
            else:
                self.iir_b = self.iir_b * (1.0 - self.iir_amount) + original * self.iir_amount
                self.ratio_b = self.ratio_b * (1.0 - self.speed) + sense * self.speed
                self.ratio_b = min(self.ratio_b, self.depth)
                ratio = self.ratio_b
                processed = self.iir_b + (original - self.iir_b) / ratio if ratio > 1.0 else original
            self.flip = not self.flip
            output[index] = processed
            removed[index] = original - processed
            ratios[index] = ratio

        active = ratios > 1.0001
        reduction = 20.0 * np.log10(ratios[active]) if np.any(active) else np.array([0.0])
        return output, {
            "active_percent": float(np.mean(active) * 100.0),
            "p95_reduction_db": float(np.percentile(reduction, 95)),
            "max_reduction_db": float(np.max(reduction)),
            "removed_rms_dbfs": float(20.0 * np.log10(np.sqrt(np.mean(removed**2)) + 1e-15)),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bands-json", default="[]")
    parser.add_argument("--deesser-frequency", type=float)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    bands = json.loads(args.bands_json)
    highpassed = highpass_80hz(mono, sample_rate)

    # Calibrate detector drive per clip so different recording levels receive
    # the same restrained studio compression instead of a fixed threshold.
    low_drive_db, high_drive_db = -12.0, 6.0
    target_p95_db = 1.8
    for _ in range(7):
        trial_drive_db = (low_drive_db + high_drive_db) / 2.0
        trial_drive = math.pow(10.0, trial_drive_db / 20.0)
        _, trial_stats = ButterComp2(sample_rate, 1.0, 0.85).process(highpassed * trial_drive)
        if trial_stats["p95_reduction_db"] > target_p95_db:
            high_drive_db = trial_drive_db
        else:
            low_drive_db = trial_drive_db
    drive_db = (low_drive_db + high_drive_db) / 2.0
    drive = math.pow(10.0, drive_db / 20.0)
    compressed, comp_stats = ButterComp2(sample_rate, 1.0, 0.85).process(highpassed * drive)
    compressed /= drive

    equalized = transparent_eq(compressed, sample_rate, bands)
    if args.deesser_frequency is not None:
        deessed, deess_stats = DeBess(sample_rate, args.deesser_frequency).process(equalized)
        # A light lower-presence stage controls the hard 4–5 kHz edge that a
        # higher automatically detected S center can leave untouched.
        deessed, lower_deess_stats = DeBess(
            sample_rate, 4500.0, intensity=0.36, depth_control=0.82
        ).process(deessed)
        # A separate, gentler air-band stage catches whistle above 10 kHz
        # without forcing the main consonant detector to lisp the whole S.
        deessed, upper_deess_stats = DeBess(
            sample_rate, 10000.0, intensity=0.35, depth_control=0.80
        ).process(deessed)
    else:
        deessed, deess_stats, lower_deess_stats, upper_deess_stats = equalized, None, None, None
    sf.write(args.output, np.clip(deessed, -1.0, 1.0), sample_rate, subtype="FLOAT")

    print(
        f"ButterComp2 gain reduction: adaptive drive {drive_db:+.2f} dB, "
        f"median {comp_stats['median_reduction_db']:.2f} dB, "
        f"p95 {comp_stats['p95_reduction_db']:.2f} dB, "
        f"max {comp_stats['max_reduction_db']:.2f} dB"
    )
    if deess_stats is not None:
        print(
            f"Airwindows DeBess: active {deess_stats['active_percent']:.1f}%, "
            f"p95 {deess_stats['p95_reduction_db']:.2f} dB, "
            f"max {deess_stats['max_reduction_db']:.2f} dB, "
            f"removed floor {deess_stats['removed_rms_dbfs']:.1f} dBFS"
        )
        print(
            f"Airwindows DeBess lower-presence: active {lower_deess_stats['active_percent']:.1f}%, "
            f"p95 {lower_deess_stats['p95_reduction_db']:.2f} dB, "
            f"max {lower_deess_stats['max_reduction_db']:.2f} dB, "
            f"removed floor {lower_deess_stats['removed_rms_dbfs']:.1f} dBFS"
        )
        print(
            f"Airwindows DeBess upper-air: active {upper_deess_stats['active_percent']:.1f}%, "
            f"p95 {upper_deess_stats['p95_reduction_db']:.2f} dB, "
            f"max {upper_deess_stats['max_reduction_db']:.2f} dB, "
            f"removed floor {upper_deess_stats['removed_rms_dbfs']:.1f} dBFS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
