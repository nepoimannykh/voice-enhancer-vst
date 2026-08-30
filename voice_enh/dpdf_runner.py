"""Stable process wrapper for DPDFNet without its crashing progress UI."""
from __future__ import annotations

import argparse
from pathlib import Path

from dpdfnet.api import enhance_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--attn-limit-db", type=float)
    args = parser.parse_args()
    enhance_file(
        input_path=args.input,
        output_path=args.output,
        model="dpdfnet8_48khz_hr",
        attn_limit_db=args.attn_limit_db,
        progress_callback=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
