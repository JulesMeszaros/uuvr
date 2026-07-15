This project is a fork of the [[https://github.com/seanghay/uvr]](UVR CLI) project by seanghay.

---

# Ultimate Ultimate Vocal Remover (uuvr)

[[Colab]](https://colab.research.google.com/drive/1VDncdndceKanFrs2LU-LM4Odv8tnPkzD?usp=sharing)

A command-line tool for separating an audio file into instrumental and vocal stems, using a pretrained `CascadedASPPNet` model (PyTorch). It's a thin inference wrapper around the `uvr5_pack` library extracted from the UVR5 project — no training code, just separation.

`separate.py` holds the core inference logic (`_audio_pre_`), and `cli.py` is the command-line entry point that wires arguments to it. You can point it at a single audio file or a whole folder, and choose where the instrumental/vocal tracks get written.

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

### 2. Python dependencies

```shell
pip install -r requirements.txt
```

`requirements.txt` does not pin `torch`/`torchaudio` — install the build that matches your hardware by following the [PyTorch install instructions](https://pytorch.org/get-started/locally/) (CUDA, MPS/Apple Silicon, or CPU-only).

The device is auto-detected at runtime (`cuda` > `mps` > `cpu`, see `--device` in [Usage](#usage)) — just make sure the PyTorch build you install actually supports the hardware you want to use.

### 3. Model weights

```shell
./download.sh
```

Fetches `uvr5_weights/2_HP-UVR.pth` if it isn't already present.

## Usage

```shell
python cli.py <audio_path> [-o OUTPUT] [--vocal-dir NAME] [--instrumental-dir NAME] [-f FORMAT]
python cli.py --input-dir <folder> [-o OUTPUT] [--vocal-dir NAME] [--instrumental-dir NAME] [-f FORMAT]
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
python cli.py 360.aac -o out --format flac
python cli.py --input-dir my_songs --vocal-dir vocals --instrumental-dir instru
python cli.py 360.aac --device cpu
```

`is_half` and `model_path` are still hardcoded at the top of `cli.py`'s `main()` — edit them directly to change model/precision.
