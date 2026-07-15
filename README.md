This project is a fork of the [UVR CLI](https://github.com/seanghay/uvr) project by seanghay.

---

# Ultimate Ultimate Vocal Remover (uuvr)

A command-line tool that separates an audio file into instrumental and vocal stems.

- Process a single file, or a whole folder at once with `--input-dir`
- Reads `.wav`, `.mp3`, `.flac`, `.aac`, `.m4a`, `.ogg`; writes output as `wav`, `flac`, `mp3`, `ogg`, or `m4a`
- Auto-detects the best available device (CUDA, Apple Silicon MPS, or CPU) — no manual config needed
- Model weights download automatically on first run and are cached locally
- Choose where the instrumental and vocal tracks get written, with optional separate subfolders for each

## Installation

### 1. System dependencies

- **[`ffmpeg`](https://ffmpeg.org/download.html)** — used both for decoding input audio and, when `--format` isn't `wav`, for encoding the output.
- **[`libsndfile`](https://github.com/libsndfile/libsndfile)** — required by `soundfile`/`librosa` to read audio files.

On macOS with Homebrew:

```shell
brew install ffmpeg libsndfile
```

On Debian/Ubuntu:

```shell
sudo apt install ffmpeg libsndfile1
```

### 2. Install the package

```shell
pip install -e .
# or, without cloning:
pip install git+https://github.com/JulesMeszaros/uuvr
```

This does not pin `torch`/`torchaudio` — install the build that matches your hardware by following the [PyTorch install instructions](https://pytorch.org/get-started/locally/) (CUDA, MPS/Apple Silicon, or CPU-only).

The device is auto-detected at runtime (`cuda` > `mps` > `cpu`, see `--device` in [Usage](#usage)) — just make sure the PyTorch build you install actually supports the hardware you want to use.

### 3. Model weights

Downloaded automatically to `~/.cache/uuvr/2_HP-UVR.pth` the first time you run `uuvr` — no manual step needed.

## Usage

```shell
uuvr <audio_path> [-o OUTPUT] [--vocal-dir NAME] [--instrumental-dir NAME] [-f FORMAT]
uuvr --input-dir <folder> [-o OUTPUT] [--vocal-dir NAME] [--instrumental-dir NAME] [-f FORMAT]
```

| Argument | Description |
| --- | --- |
| `audio_path` | Single audio file to separate. Mutually exclusive with `--input-dir`. |
| `--input-dir <folder>` | Batch-process every audio file (`.wav`, `.mp3`, `.flac`, `.aac`, `.m4a`, `.ogg`) found in this folder. Mutually exclusive with `audio_path`. |
| `-o, --output <dir>` | Output directory (default: `out`). |
| `--vocal-dir <name>` | Optional subfolder name, created inside `--output`, for the vocal track. Omit to write it directly into `--output`. |
| `--instrumental-dir <name>` | Optional subfolder name, created inside `--output`, for the instrumental track. Omit to write it directly into `--output`. |
| `-f, --format <fmt>` | Output audio format: `wav` (default), `flac`, `mp3`, `ogg`, or `m4a`. Anything other than `wav` requires `ffmpeg`. |
| `--device <dev>` | `auto` (default, picks `cuda` > `mps` > `cpu`), or force `cuda`/`mps`/`cpu`. |

Output files are named `instrument_<name>.<format>` and `vocal_<name>.<format>`.

Examples:

```shell
uuvr 360.aac -o out --format flac
uuvr --input-dir my_songs --vocal-dir vocals --instrumental-dir instru
uuvr 360.aac --device cpu
```

`is_half` is still hardcoded at the top of `cli.py`'s `main()` — edit it directly to change precision.
