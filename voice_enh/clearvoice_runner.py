"""Small process wrapper for ClearerVoice's MossFormer2 48 kHz model."""
from __future__ import annotations

import argparse
from pathlib import Path

from clearvoice import ClearVoice


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    model = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    enhanced = model(input_path=str(args.input), online_write=False)
    model.write(enhanced, output_path=str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
