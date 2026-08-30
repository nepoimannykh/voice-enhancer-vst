from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__


PRESETS = ("natural", "studio", "podcast")

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
    p.add_argument(
        "--dereverb-attn-limit-db", type=float,
        help="limit DPDFNet dereverberation in dB; higher values apply more enhancement",
    )
    p.add_argument("--resolve", action="store_true",
                   help="process the input file in place for DaVinci Resolve External Audio Process")
    p.add_argument("--deesser", action=argparse.BooleanOptionalAction, default=True,
                   help="adaptive sibilance detector/de-esser (default: on)")
    p.add_argument(
        "--adaptive-eq", action=argparse.BooleanOptionalAction, default=True,
        help="calculate broad U87 tonal matching from the processed clip (default: on)",
    )
    p.add_argument(
        "--eq-reference", type=Path,
        help="optional clean speech reference used instead of the bundled U87 spectral profile",
    )
    p.add_argument(
        "--compressor", choices=("buttercomp", "ffmpeg", "none"), default="buttercomp",
        help="studio ButterComp2 (default), legacy FFmpeg compressor, or no compression",
    )
    p.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True, help="produce mono voice audio (default: on)")
    p.add_argument("--sample-rate", type=int, default=48000, choices=(44100, 48000))
    p.add_argument("-f", "--force", action="store_true", help="overwrite an existing output")
    p.add_argument("--dry-run", action="store_true", help="print commands without processing")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def base_filters(
    deesser: bool = True, tonal_eq: list[str] | None = None, deesser_frequency: int | None = None
) -> list[str]:
    filters = [
        "highpass=f=80:p=2",
    ]
    filters.extend(([
        "equalizer=f=110:t=q:w=0.7:g=-2",
        "equalizer=f=165:t=q:w=0.65:g=3",
        "equalizer=f=380:t=q:w=0.55:g=-2.5",
        "equalizer=f=900:t=q:w=0.45:g=2",
        "equalizer=f=2500:t=q:w=0.5:g=-1",
        "equalizer=f=4200:t=q:w=0.55:g=-2",
        "equalizer=f=6800:t=q:w=0.45:g=1.5",
    ] if tonal_eq is None else tonal_eq))
    if deesser:
        # Center detection is clip-adaptive; stronger engagement catches the
        # iPhone whistle while the reduction cap limits lisping.
        normalized_frequency = 0.42 if deesser_frequency is None else deesser_frequency / 24000.0
        filters.append(f"deesser=i=0.45:m=0.30:f={normalized_frequency:.4f}")
    return filters


def describe_dereverb(attn_limit_db: float | None) -> str:
    if attn_limit_db is None:
        return "full: unlimited, 100% enhanced signal"
    enhanced = (1.0 - math.pow(10.0, -attn_limit_db / 20.0)) * 100.0
    if attn_limit_db == 0:
        level = "off"
    elif attn_limit_db <= 3:
        level = "light"
    elif attn_limit_db <= 6:
        level = "moderate"
    elif attn_limit_db <= 9:
        level = "strong"
    else:
        level = "aggressive"
    return f"{level}: {attn_limit_db:g} dB limit, {enhanced:.0f}% enhanced blend"


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
    command = [neural, option, str(work), "--atten-lim", "9", str(model_input)]
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


def dpdf_enhance(dpdf: str, source: Path, destination: Path, attn_limit_db: float | None = None) -> Path:
    command = [sys.executable, "-m", "voice_enh.dpdf_runner", str(source), str(destination)]
    if attn_limit_db is not None:
        command.extend(["--attn-limit-db", str(attn_limit_db)])
    result = run(command, capture=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "DPDFNet failed")
    if not destination.is_file():
        raise RuntimeError("DPDFNet did not create an output file")
    return destination


def buttercomp_enhance(source: Path, destination: Path) -> str:
    result = run(
        [
            sys.executable, "-m", "voice_enh.buttercomp_runner", str(source), str(destination),
            "--compress", "1.0", "--wet", "0.85", "--drive-db", "3",
        ],
        capture=True,
    )
    if result.returncode or not destination.is_file():
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ButterComp2 failed")
    return result.stdout.strip()


def studio_dsp_enhance(source: Path, destination: Path, bands: list, deesser_frequency: int | None) -> str:
    payload = json.dumps([
        {"frequency": band.frequency, "q": band.q, "gain_db": band.gain_db}
        for band in bands
        if band.gain_db != 0
    ])
    command = [
        sys.executable, "-m", "voice_enh.studio_dsp_runner", str(source), str(destination),
        "--bands-json", payload,
    ]
    if deesser_frequency is not None:
        command.extend(["--deesser-frequency", str(deesser_frequency)])
    result = run(command, capture=True)
    if result.returncode or not destination.is_file():
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "studio DSP failed")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("error: FFmpeg is required but was not found in PATH", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    if args.dereverb_attn_limit_db is not None and args.dereverb_attn_limit_db < 0:
        print("error: --dereverb-attn-limit-db must be non-negative", file=sys.stderr)
        return 2
    if args.eq_reference is not None and not args.eq_reference.is_file():
        print(f"error: EQ reference does not exist: {args.eq_reference}", file=sys.stderr)
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
        filters = base_filters(args.deesser)
        codec = audio_args(output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    channel_args = ["-ac", "1"] if args.mono else []
    if args.dry_run:
        if args.engine == "neural":
            print("DeepFilterNet: convert input to mono 48 kHz WAV, run neural enhancement")
        print("Restore the source's measured LUFS loudness")
        print(f"Compressor: {args.compressor}")
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

    print(f"Measuring source loudness for exact matching…", file=sys.stderr)
    try:
        source_stats = read_loudness(ffmpeg, args.input)
        source_lufs = float(source_stats["input_i"])
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    temp = tempfile.TemporaryDirectory(prefix="voice-enh-")
    try:
        processing_source = args.input
        adaptive_bands = None
        deesser_frequency = None
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
                strength = args.dereverb_attn_limit_db
                print(f"Running DPDFNet dereverb ({describe_dereverb(strength)})…", file=sys.stderr)
                processing_source = dpdf_enhance(
                    dpdf, processing_source, Path(temp.name) / "dpdf-enhanced.wav", strength
                )
            print("Enhancing with ClearVoice MossFormer2 48 kHz (one pass)…", file=sys.stderr)
            processing_source = clearvoice_enhance(processing_source, Path(temp.name) / "clearvoice-enhanced.wav")
            print("Preserving the natural room floor (no second cleanup pass)…", file=sys.stderr)

        if args.adaptive_eq:
            from .spectral_match import adaptive_u87_bands, describe_bands, ffmpeg_filters, sibilance_center

            print("Calculating adaptive U87 spectral-envelope match…", file=sys.stderr)
            bands = adaptive_u87_bands(processing_source, args.eq_reference)
            adaptive_bands = bands
            tonal_eq = ffmpeg_filters(bands)
            deesser_frequency = sibilance_center(processing_source) if args.deesser else None
            filters = base_filters(args.deesser, tonal_eq, deesser_frequency)
            descriptions = describe_bands(bands)
            if descriptions:
                for description in descriptions:
                    print(f"  EQ {description}", file=sys.stderr)
            else:
                print("  EQ no confident correction required", file=sys.stderr)
            if deesser_frequency is not None:
                print(f"  De-esser detected S center: {deesser_frequency / 1000:g} kHz", file=sys.stderr)
        else:
            print("Using fixed broad U87 tonal profile…", file=sys.stderr)
            filters = base_filters(args.deesser)

        if args.compressor == "buttercomp":
            postfiltered = Path(temp.name) / "postfiltered.wav"
            prepare_command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(processing_source),
                "-map", "0:a:0", "-vn",
            ]
            if filters and adaptive_bands is None:
                prepare_command.extend(["-af", ",".join(filters)])
            prepare_command.extend(["-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", str(postfiltered)])
            prepared = run(prepare_command, capture=True)
            if prepared.returncode:
                raise RuntimeError(prepared.stderr.strip() or "could not prepare audio for ButterComp2")
            processing_source = Path(temp.name) / "buttercomp.wav"
            if adaptive_bands is not None:
                print(
                    "Processing 80 Hz high-pass → Airwindows ButterComp2 → "
                    "double-precision studio EQ → Airwindows DeBess…",
                    file=sys.stderr,
                )
                compression_log = studio_dsp_enhance(
                    postfiltered, processing_source, adaptive_bands,
                    deesser_frequency if args.deesser else None,
                )
            else:
                print("Compressing with Airwindows ButterComp2 (85% parallel blend)…", file=sys.stderr)
                compression_log = buttercomp_enhance(postfiltered, processing_source)
            if compression_log:
                print(f"  {compression_log}", file=sys.stderr)
            filters = []
        elif args.compressor == "ffmpeg":
            print("Compressing with legacy FFmpeg compressor…", file=sys.stderr)
            filters.append("acompressor=threshold=0.20:ratio=1.3:attack=60:release=180:makeup=1:knee=3")
        else:
            print("Compression disabled…", file=sys.stderr)

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
        temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
