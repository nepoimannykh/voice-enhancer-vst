from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__


PRESETS = ("natural", "studio", "podcast")

TARGETS = {
    "natural": (-18.0, 9.0, -1.5),
    "studio": (-16.0, 8.0, -1.0),
    "podcast": (-16.0, 7.0, -1.0),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice-enh",
        description="Enhance speech recordings with DeepFilterNet neural speech enhancement.",
    )
    p.add_argument("input", type=Path, help="input audio or video file")
    p.add_argument("-o", "--output", type=Path, help="output path (default: <input>-enhanced.wav)")
    p.add_argument("-p", "--preset", choices=PRESETS, default="studio")
    p.add_argument(
        "--engine", choices=("neural", "classic"), default="neural",
        help="DeepFilterNet neural enhancement (default) or FFmpeg-only cleanup",
    )
    p.add_argument("--dereverb", action=argparse.BooleanOptionalAction, default=True,
                   help="run aggressive DPDFNet dereverb before DeepFilterNet (default: on)")
    p.add_argument("--voicefixer", action=argparse.BooleanOptionalAction, default=True,
                   help="run VoiceFixer neural restoration after MossFormer2 (default: on)")
    p.add_argument("--resolve", action="store_true",
                   help="process the input file in place for DaVinci Resolve External Audio Process")
    p.add_argument("--deesser", action=argparse.BooleanOptionalAction, default=True,
                   help="adaptive sibilance detector/de-esser (default: on)")
    p.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True, help="produce mono voice audio (default: on)")
    p.add_argument("--sample-rate", type=int, default=48000, choices=(44100, 48000))
    p.add_argument("-f", "--force", action="store_true", help="overwrite an existing output")
    p.add_argument("--dry-run", action="store_true", help="print commands without processing")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def base_filters(preset: str, noise_reduction: float | None) -> list[str]:
    return []


def loudnorm_filter(preset: str, measured: dict[str, str] | None = None) -> str:
    loudness, range_, peak = TARGETS[preset]
    value = f"loudnorm=I={loudness}:LRA={range_}:TP={peak}"
    if measured:
        value += (
            f":measured_I={measured['input_i']}:measured_LRA={measured['input_lra']}"
            f":measured_TP={measured['input_tp']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true"
        )
    return value


def audio_args(output: Path) -> list[str]:
    suffix = output.suffix.lower()
    if suffix == ".wav":
        return ["-c:a", "pcm_s24le"]
    if suffix == ".flac":
        return ["-c:a", "flac"]
    if suffix == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", "192k"]
    if suffix in {".m4a", ".aac"}:
        return ["-c:a", "aac", "-b:a", "192k"]
    raise ValueError("output must end in .wav, .flac, .mp3, .m4a, or .aac")


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=capture, check=False)


def find_neural_executable() -> str | None:
    """Find DeepFilterNet in PATH or beside the active virtualenv Python."""
    found = shutil.which("deepFilter") or shutil.which("deep-filter")
    if found:
        return found
    scripts = Path(sys.executable).parent
    for name in ("deepFilter", "deep-filter"):
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    return None


def clearvoice_enhance(source: Path, destination: Path) -> Path:
    result = run([sys.executable, "-m", "voice_enh.clearvoice_runner", str(source), str(destination)], capture=True)
    if result.returncode or not destination.is_file():
        raise RuntimeError(result.stderr.strip() or "ClearVoice enhancement failed")
    return destination


def voicefixer_enhance(source: Path, destination: Path) -> Path:
    result = run([sys.executable, "-m", "voice_enh.voicefixer_runner", str(source), str(destination)], capture=True)
    if result.returncode or not destination.is_file():
        raise RuntimeError(result.stderr.strip() or "VoiceFixer enhancement failed")
    return destination


def find_dpdf_executable() -> str | None:
    found = shutil.which("dpdfnet")
    if found:
        return found
    candidate = Path(sys.executable).parent / "dpdfnet"
    return str(candidate) if candidate.is_file() else None


def read_loudness(
    ffmpeg: str, source: Path, filters: list[str] | None = None, target_i: float = -24.0
) -> dict[str, str]:
    """Measure EBU R128 loudness, optionally after a filter chain."""
    target_i = max(-70.0, min(-5.0, target_i))
    chain = list(filters or []) + [f"loudnorm=I={target_i:g}:LRA=7:TP=-1:print_format=json"]
    result = run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(source), "-map", "0:a:0", "-af", ",".join(chain), "-f", "null", "-"],
        capture=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg loudness analysis failed")
    start, end = result.stderr.rfind("{"), result.stderr.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("FFmpeg did not return loudness measurements")
    return json.loads(result.stderr[start : end + 1])


def matched_loudnorm(target_i: float, measured: dict[str, str] | None = None) -> str:
    # loudnorm supports -70..-5 LUFS. Keeping true peak below -1 dB prevents
    # loudness restoration from introducing clipping.
    target_i = max(-70.0, min(-5.0, target_i))
    value = f"loudnorm=I={target_i:g}:LRA=7:TP=-1"
    if measured:
        value += (
            f":measured_I={measured['input_i']}:measured_LRA={measured['input_lra']}"
            f":measured_TP={measured['input_tp']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true"
        )
    return value


def neural_enhance(neural: str, ffmpeg: str, source: Path, work: Path) -> Path:
    """Convert to the model contract, run DeepFilterNet, and locate its output."""
    model_input = work / "model-input.wav"
    if source != model_input:
        converted = run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(model_input),
        ], capture=True)
        if converted.returncode:
            raise RuntimeError(converted.stderr.strip() or "could not prepare audio for DeepFilterNet")
    option = "--out-dir" if Path(neural).name == "deep-filter" else "--output-dir"
    # A modest attenuation ceiling preserves room tails and avoids the
    # over-dereverberated sound that unrestricted neural suppression can cause.
    command = [neural, option, str(work), "--atten-lim", "12", str(model_input)]
    bundled_model = Path(__file__).resolve().parent.parent / ".models" / "DeepFilterNet3"
    if bundled_model.is_dir():
        command[1:1] = ["--model-base-dir", str(bundled_model)]
    if Path(neural).name == "deep-filter":
        command.insert(1, "--compensate-delay")
    before = set(work.glob("*.wav"))
    result = run(command, capture=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "DeepFilterNet failed")
    candidates = [path for path in work.glob("*.wav") if path != model_input]
    if not candidates:
        raise RuntimeError("could not identify DeepFilterNet output")
    # DeepFilterNet reuses its model-suffixed filename on a second pass.
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def dpdf_enhance(dpdf: str, source: Path, destination: Path) -> Path:
    # Omit --attn-limit-db: DPDFNet then returns 100% enhanced signal. A value
    # of 0 blends 100% of the delayed noisy reference back into the output.
    result = run([dpdf, "enhance", str(source), str(destination), "--model", "dpdfnet8_48khz_hr"], capture=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "DPDFNet failed")
    if not destination.is_file():
        raise RuntimeError("DPDFNet did not create an output file")
    return destination


def measure(ffmpeg: str, source: Path, filters: list[str], preset: str) -> dict[str, str]:
    chain = ",".join(filters + [loudnorm_filter(preset) + ":print_format=json"])
    result = run([ffmpeg, "-hide_banner", "-nostats", "-i", str(source), "-map", "0:a:0", "-af", chain, "-f", "null", "-"], capture=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg analysis failed")
    start, end = result.stderr.rfind("{"), result.stderr.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("FFmpeg did not return loudness measurements")
    return json.loads(result.stderr[start : end + 1])


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("error: FFmpeg is required but was not found in PATH", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    output = args.output or args.input.with_name(f"{args.input.stem}-enhanced.wav")
    if args.resolve:
        output = args.input.with_name(f".{args.input.stem}.voice-enh-resolve.wav")
    if output.resolve() == args.input.resolve() and not args.resolve:
        print("error: output must differ from input", file=sys.stderr)
        return 2
    if output.exists() and not (args.force or args.resolve):
        print(f"error: output already exists: {output} (use --force to overwrite)", file=sys.stderr)
        return 2
    try:
        filters = [
            "highpass=f=80:p=2",
            "adynamicequalizer=threshold=8:dfrequency=120:dqfactor=0.8:tfrequency=110:tqfactor=0.8:attack=2:release=80:ratio=8:range=12:auto=adaptive:mode=cutabove",
            "adeclick=w=35:o=60:t=6:b=1",
            "agate=threshold=0.015:range=0.35:ratio=2:attack=10:release=180:makeup=1:detection=rms",
            "equalizer=f=220:t=q:w=0.9:g=-2.5",
            "acompressor=threshold=0.10:ratio=2.5:attack=25:release=150:makeup=1:knee=3",
            "deesser=i=0.6:m=0.9:f=0.5",
        ] if args.deesser else ["acompressor=threshold=0.10:ratio=3:attack=8:release=120:makeup=1:knee=3"]
        codec = audio_args(output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    channel_args = ["-ac", "1"] if args.mono else []
    if args.dry_run:
        if args.engine == "neural":
            print("DeepFilterNet: convert input to mono 48 kHz WAV, run neural enhancement")
        print("Restore the source's measured LUFS loudness")
        print(f"Output: {output}")
        return 0

    neural = find_neural_executable()
    dpdf = find_dpdf_executable() if args.dereverb else None
    if args.engine == "neural" and not neural:
        print(
            "error: neural mode requires DeepFilterNet. Install it with:\n"
            "  python3 -m pip install deepfilternet\n"
            "or use --engine classic.",
            file=sys.stderr,
        )
        return 2
    if args.engine == "neural" and args.dereverb and not dpdf:
        print("error: install DPDFNet with .venv/bin/python -m pip install dpdfnet or use --no-dereverb", file=sys.stderr)
        return 2
    if args.engine == "neural" and args.voicefixer:
        try:
            import voicefixer  # noqa: F401
        except ImportError:
            print("error: install VoiceFixer with .venv/bin/python -m pip install voicefixer or use --no-voicefixer", file=sys.stderr)
            return 2

    print(f"Measuring source loudness for exact matching…", file=sys.stderr)
    try:
        source_stats = read_loudness(ffmpeg, args.input)
        source_lufs = float(source_stats["input_i"])
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    temp = tempfile.TemporaryDirectory(prefix="voice-enh-") if args.engine == "neural" else None
    try:
        processing_source = args.input
        if args.engine == "neural":
            # Prepare once at the model contract (mono, 48 kHz), then run the
            # high-impact dual-path stage before DeepFilterNet's final pass.
            model_input = Path(temp.name) / "model-input.wav"
            prepared = run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.input),
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(model_input),
            ], capture=True)
            if prepared.returncode:
                raise RuntimeError(prepared.stderr.strip() or "could not prepare audio for neural enhancement")
            print("Suppressing noise with DeepFilterNet3…", file=sys.stderr)
            processing_source = neural_enhance(neural, ffmpeg, model_input, Path(temp.name))
            if args.dereverb:
                print("Running aggressive DPDFNet dereverb…", file=sys.stderr)
                processing_source = dpdf_enhance(dpdf, processing_source, Path(temp.name) / "dpdf-enhanced.wav")
            print("Enhancing with ClearVoice MossFormer2 48 kHz (aggressive pass 1)…", file=sys.stderr)
            processing_source = clearvoice_enhance(processing_source, Path(temp.name) / "clearvoice-enhanced.wav")
            if args.voicefixer:
                print("Restoring speech with VoiceFixer neural vocoder…", file=sys.stderr)
                processing_source = voicefixer_enhance(processing_source, Path(temp.name) / "voicefixer-enhanced.wav")
            if args.dereverb:
                print("Finishing with DeepFilterNet3 voice cleanup…", file=sys.stderr)
                processing_source = neural_enhance(neural, ffmpeg, processing_source, Path(temp.name))
            filters = [
                "highpass=f=80:p=2",
                "adynamicequalizer=threshold=8:dfrequency=120:dqfactor=0.8:tfrequency=110:tqfactor=0.8:attack=2:release=80:ratio=8:range=12:auto=adaptive:mode=cutabove",
                # Conservative click repair: avoid smearing speech transients.
                "adeclick=w=35:o=60:t=6:b=1",
                "agate=threshold=0.015:range=0.35:ratio=2:attack=10:release=180:makeup=1:detection=rms",
                "equalizer=f=220:t=q:w=0.9:g=-2.5",
                "acompressor=threshold=0.10:ratio=2.5:attack=25:release=150:makeup=1:knee=3",
            ]
            if args.deesser:
                filters.append("deesser=i=0.6:m=0.9:f=0.5")

        print(
            f"Restoring source loudness at {source_lufs:.1f} LUFS…",
            file=sys.stderr,
        )
        measured = read_loudness(ffmpeg, processing_source, filters, source_lufs)
        final_chain = ",".join(filters + [matched_loudnorm(source_lufs, measured)])
        common = [
            ffmpeg, "-hide_banner", "-y" if args.force else "-n", "-i", str(processing_source),
            "-map", "0:a:0", "-vn",
        ]
        command = common + ["-af", final_chain] + channel_args + ["-ar", str(args.sample_rate)] + codec + [str(output)]
        result = run(command)
        if result.returncode:
            print("error: FFmpeg processing failed", file=sys.stderr)
            return result.returncode
        if args.resolve:
            os.replace(output, args.input)
            print(f"Updated {args.input} (matched to {source_lufs:.1f} LUFS)")
        else:
            print(f"Created {output} (matched to {source_lufs:.1f} LUFS)")
        return 0
    except (RuntimeError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
