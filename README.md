# voice-enh

`voice-enh` enhances AirPods, Mac microphone, phone, and laptop speech recordings. It converts audio to 48 kHz mono, applies neural noise suppression, dereverberation, and speech restoration, then matches the source's measured LUFS loudness.

## Requirements and installation

- Python 3.12
- [FFmpeg](https://ffmpeg.org/) in `PATH`

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural]'
```

Models download automatically on first use.

Allow approximately **2 GB of free disk space**: about 1.1 GB for the Python/ML environment, 230 MB for bundled/downloaded model weights, and extra temporary space while processing. Individual WAV outputs require roughly 7 MB per minute at 48 kHz mono 24-bit.

## Usage

```sh
voice-enh input.m4a -o output.wav --force
```

The default output is mono, 48 kHz, 24-bit WAV. Use `voice-enh --help` for options.

## Processing

- [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) — neural noise suppression
- [DPDFNet](https://github.com/ceva-ip/DPDFNet) — dual-path dereverberation
- [ClearVoice MossFormer2 SE 48K](https://github.com/modelscope/ClearerVoice-Studio) — full-band speech enhancement
- [FFmpeg compressor](https://ffmpeg.org/ffmpeg-filters.html#acompressor) — podcast dynamics
- [FFmpeg de-esser](https://ffmpeg.org/ffmpeg-filters.html#deesser) — sibilance reduction
- [FFmpeg loudnorm](https://ffmpeg.org/ffmpeg-filters.html#loudnorm) — source LUFS matching

## DaVinci Resolve

The repository includes `voice-enh-resolve.command`, a directly executable launcher for Resolve. Use this script with Resolve’s **Command Line** mode; Resolve appends the bounced WAV path as an argument. The `.app` launchers are only suitable for Reveal/Clipboard workflows and may receive no command-line arguments.

1. Open **DaVinci Resolve > Preferences > System > Audio Plugins**.
2. Under **Setup External Audio Processes**, click **Add**.
3. Name it `voice-enh`.
4. Set **Path** to `/Users/jenya/IdeaProjects/2026-2/voice-enh/voice-enh-resolve.command`. If it is not shown, press **Command-Shift-G** in the file dialog and paste that exact path.
5. Set **Type** to **Command Line**, save, and restart Resolve if prompted.
6. In Fairlight, right-click a clip and choose **External Audio Process > voice-enh**.

Resolve keeps the original clip and imports the processed result as a new layer. To diagnose a run, inspect `/tmp/voice-enh-resolve.log`; a successful invocation contains the bounced file in `args:`/`input:` followed by `exit: 0`.
