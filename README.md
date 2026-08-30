# voice-enh

`voice-enh` enhances AirPods, Mac microphone, phone, and laptop speech recordings. It converts audio to 48 kHz mono, applies one restrained noise-suppression pass, dereverberation, and one speech-restoration pass. Broad reference EQ, selective de-essing, gentle compression, and source-LUFS matching follow the neural stages.

## DaVinci Resolve integration

After completing the installation below, connect the included launcher to Resolve:

1. Open **DaVinci Resolve > Preferences > System > Audio Plugins**.
2. Under **Setup External Audio Processes**, click **Add**.
3. Name the process `voice-enh`.
4. Set **Path** to `/Users/jenya/IdeaProjects/2026-2/voice-enh/voice-enh-resolve.command`. If the file is not visible, press **Command-Shift-G** and paste the path.
5. Set **Type** to **Command Line**, save, and restart Resolve if prompted.
6. In Fairlight, right-click an audio clip and select **External Audio Process > voice-enh**.

Resolve keeps the original clip and imports the enhanced WAV as a new layer. The launcher requires **Command Line** mode so Resolve passes it the bounced WAV path. For diagnostics, inspect `/tmp/voice-enh-resolve.log`; a successful run ends with `exit: 0`.

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

The default output is mono, 48 kHz, 24-bit WAV. Broad U87 spectral-envelope matching and sibilance-center detection are calculated independently for every processed clip. Use `--no-adaptive-eq` for the legacy fixed profile, or `--eq-reference clean-speech.wav` to calculate the target from another clean speech recording. Use `voice-enh --help` for all options.

## Processing

After neural enhancement, the tonal/dynamics chain is deliberately ordered as **80 Hz high-pass → ButterComp2 → adaptive U87 EQ → DeBess → loudness matching**. MossFormer remains a single unchanged neural pass; the later stages only correct frequency balance, dynamics, and sibilance.

- [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) — one restrained neural noise-suppression pass; no second cleanup pass, preserving a stable natural floor
- [DPDFNet](https://github.com/ceva-ip/DPDFNet) — dual-path dereverberation
- [ClearVoice MossFormer2 SE 48K](https://github.com/modelscope/ClearerVoice-Studio) — full-band speech enhancement
- Double-precision transparent adaptive U87 EQ — RBJ parametric sections, voiced-frame spectral-envelope matching, confidence weighting, a global ±3 dB correction budget, and no high-frequency synthesis
- [Airwindows DeBess](https://github.com/airwindows/airwindows) — MIT-licensed slew-structure detection: a clip-centered main stage, a 4.5 kHz lower-presence stage capped around 1.7 dB, and a 10 kHz upper-air stage capped around 2 dB
- [Airwindows ButterComp2](https://github.com/airwindows/airwindows) — MIT-licensed, program-dependent studio compression at an 85% parallel blend; use `--compressor ffmpeg` for the legacy comparison
- [FFmpeg loudnorm](https://ffmpeg.org/ffmpeg-filters.html#loudnorm) — source LUFS matching

ButterComp2 and DeBess attribution and their MIT terms are recorded in `THIRD_PARTY_LICENSES.md`.
