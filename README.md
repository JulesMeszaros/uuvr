This project is a fork of the [[https://github.com/seanghay/uvr]](UVR CLI) project by seanghay.

---

# Ultimate Ultimate Vocal Remover (uuvr)

[[Colab]](https://colab.research.google.com/drive/1VDncdndceKanFrs2LU-LM4Odv8tnPkzD?usp=sharing)

A command-line tool for separating an audio file into instrumental and vocal stems, using a pretrained `CascadedASPPNet` model (PyTorch). It's a thin inference wrapper around the `uvr5_pack` library extracted from the UVR5 project — no training code, just separation.

`separate.py` holds the core inference logic (`_audio_pre_`), and `cli.py` is the command-line entry point that wires arguments to it. You can point it at a single audio file or a whole folder, and choose where the instrumental/vocal tracks get written.

⚠️ Before running this project, make sure you have installed `torch`, `torchaudio`. Please check out the PyTorch documentation.

⚠️ Also make sure you have `libsndfile` and `ffmpeg` installed.

⚠️ This project currently works on CUDA.

## Install dependencies

```shell
pip install -r requirements.txt
```

## Download Model Weights

```shell
./download.sh
```

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

Output files are named `instrument_<name>.<format>` and `vocal_<name>.<format>`.

Examples:

```shell
python cli.py 360.aac -o out --format flac
python cli.py --input-dir my_songs --vocal-dir vocals --instrumental-dir instru
```

`device`, `is_half`, and `model_path` are still hardcoded at the top of `cli.py`'s `main()` — edit them directly to change model/device.
