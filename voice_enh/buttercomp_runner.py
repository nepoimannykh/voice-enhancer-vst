"""Headless mono adaptation of Airwindows ButterComp2.

Original DSP copyright (c) Chris Johnson / Airwindows, MIT License.
See THIRD_PARTY_LICENSES.md. The signal equations follow ButterComp2's
double-precision processDoubleReplacing implementation; plugin hosting,
stereo duplication, denormal noise, and floating-point dither are omitted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf


class ButterComp2:
    def __init__(self, sample_rate: int, compress: float = 1.0, wet: float = 0.85):
        self.sample_rate = sample_rate
        self.compress = min(1.0, max(0.0, compress))
        self.wet = min(1.0, max(0.0, wet))
        self.control_a_pos = self.control_a_neg = 1.0
        self.control_b_pos = self.control_b_neg = 1.0
        self.target_pos = self.target_neg = 1.0
        self.last_output = 0.0
        self.flip = False

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        result = np.empty_like(audio, dtype=np.float64)
        input_gain = math.pow(10.0, (self.compress * 14.0) / 20.0)
        comp_factor = 0.012 * (self.compress / 135.0)
        output_gain = ((input_gain - 1.0) / 1.5) + 1.0
        overall_scale = self.sample_rate / 44100.0
        multipliers = np.empty(len(audio), dtype=np.float64)

        for index, dry in enumerate(audio.astype(np.float64, copy=False)):
            sample = dry * input_gain
            divisor = comp_factor / (1.0 + abs(self.last_output)) / overall_scale
            remainder = divisor
            divisor = 1.0 - divisor

            input_pos = max(sample + 1.0, 0.0)
            output_pos = min(input_pos / 2.0, 1.0)
            input_pos *= input_pos
            self.target_pos = self.target_pos * divisor + input_pos * remainder
            calc_pos = math.pow(1.0 / max(self.target_pos, 1e-30), 2)

            input_neg = max(-sample + 1.0, 0.0)
            output_neg = min(input_neg / 2.0, 1.0)
            input_neg *= input_neg
            self.target_neg = self.target_neg * divisor + input_neg * remainder
            calc_neg = math.pow(1.0 / max(self.target_neg, 1e-30), 2)

            if sample > 0.0:
                if self.flip:
                    self.control_a_pos = self.control_a_pos * divisor + calc_pos * remainder
                else:
                    self.control_b_pos = self.control_b_pos * divisor + calc_pos * remainder
            else:
                if self.flip:
                    self.control_a_neg = self.control_a_neg * divisor + calc_neg * remainder
                else:
                    self.control_b_neg = self.control_b_neg * divisor + calc_neg * remainder

            if self.flip:
                multiplier = self.control_a_pos * output_pos + self.control_a_neg * output_neg
            else:
                multiplier = self.control_b_pos * output_pos + self.control_b_neg * output_neg

            processed = sample * multiplier / output_gain
            processed = processed * self.wet + dry * (1.0 - self.wet)
            self.last_output = processed
            self.flip = not self.flip
            result[index] = processed
            multipliers[index] = max(multiplier, 1e-12)

        reduction_db = -20.0 * np.log10(multipliers)
        active = np.abs(audio) > max(float(np.max(np.abs(audio))) * 0.03, 1e-5)
        active_reduction = reduction_db[active] if np.any(active) else reduction_db
        stats = {
            "median_reduction_db": float(np.median(active_reduction)),
            "p95_reduction_db": float(np.percentile(active_reduction, 95)),
            "max_reduction_db": float(np.max(active_reduction)),
        }
        return np.clip(result, -1.0, 1.0), stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compress", type=float, default=1.0)
    parser.add_argument("--wet", type=float, default=0.85)
    parser.add_argument("--drive-db", type=float, default=3.0)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    drive = math.pow(10.0, args.drive_db / 20.0)
    processed, stats = ButterComp2(sample_rate, args.compress, args.wet).process(mono * drive)
    processed /= drive
    sf.write(args.output, processed, sample_rate, subtype="FLOAT")
    print(
        "ButterComp2 gain reduction: "
        f"drive {args.drive_db:g} dB, "
        f"median {stats['median_reduction_db']:.2f} dB, "
        f"p95 {stats['p95_reduction_db']:.2f} dB, "
        f"max {stats['max_reduction_db']:.2f} dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
