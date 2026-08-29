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
