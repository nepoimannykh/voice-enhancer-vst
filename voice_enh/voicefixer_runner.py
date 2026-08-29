"""Small offline wrapper for VoiceFixer's neural speech restoration model."""
from __future__ import annotations

import argparse
from voicefixer import VoiceFixer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    VoiceFixer().restore(input=args.input, output=args.output, cuda=False, mode=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
