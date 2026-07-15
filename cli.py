import argparse
import gc
import os

import torch

from separate import SUPPORTED_FORMATS, _audio_pre_

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sépare un fichier audio en pistes instrumentale et vocale."
    )
    parser.add_argument(
        "audio_path", nargs="?", default=None, help="Fichier audio à séparer"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Répertoire contenant des fichiers audio à séparer (tous les audios du dossier seront traités)",
    )
    parser.add_argument(
        "-o", "--output", default="out", help="Dossier de sortie (défaut: out)"
    )
    parser.add_argument(
        "--vocal-dir",
        default=None,
        help="Nom du sous-dossier pour le vocal, dans le dossier de sortie (défaut: directement dans le dossier de sortie)",
    )
    parser.add_argument(
        "--instrumental-dir",
        default=None,
        help="Nom du sous-dossier pour l'instrumental, dans le dossier de sortie (défaut: directement dans le dossier de sortie)",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="wav",
        choices=SUPPORTED_FORMATS,
        help="Format des fichiers de sortie (défaut: wav)",
    )
    args = parser.parse_args()
    if not args.audio_path and not args.input_dir:
        parser.error("il faut fournir audio_path ou --input-dir")
    if args.audio_path and args.input_dir:
        parser.error("audio_path et --input-dir sont exclusifs, choisis un seul")
    return args


def main():
    device = "mps"
    is_half = False
    model_path = "uvr5_weights/2_HP-UVR.pth"
    args = _parse_args()
    vocal_root = (
        os.path.join(args.output, args.vocal_dir) if args.vocal_dir else args.output
    )
    instrumental_root = (
        os.path.join(args.output, args.instrumental_dir)
        if args.instrumental_dir
        else args.output
    )
    if args.input_dir:
        audio_paths = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(AUDIO_EXTENSIONS)
        )
    else:
        audio_paths = [args.audio_path]

    pre_fun = _audio_pre_(model_path=model_path, device=device, is_half=is_half)
    for audio_path in audio_paths:
        pre_fun._path_audio_(
            audio_path, instrumental_root, vocal_root, format=args.format
        )
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
