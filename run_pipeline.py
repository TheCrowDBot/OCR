"""
Pipeline completa: export do Label Studio + crop -> geração de dados sintéticos
-> treino do modelo CRNN com validação cruzada.

Uso:
    python run_pipeline.py                          # corre tudo
    python run_pipeline.py --skip-synthetic          # sem passo de fontes sintéticas
    python run_pipeline.py --skip-export --skip-synthetic   # só treino
    python run_pipeline.py --fonts fonts --variations 3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name: str, args: list[str]) -> None:
    print(f"\n{'#' * 60}")
    print(f"#  {name}")
    print(f"{'#' * 60}\n")
    result = subprocess.run(args, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n'{name}' falhou (exit code {result.returncode}). A parar a pipeline.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Corre a pipeline completa de treino OCR.")
    parser.add_argument("--skip-export", action="store_true",
                        help="Não corre export_and_crop.py (usa real_crops/ já existente)")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Não corre generate_words_fonts_v3.py (sem dados sintéticos)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Não corre train.py")
    parser.add_argument("--fonts", default="fonts",
                        help="Pasta de fontes .ttf para generate_words_fonts_v3.py (default: fonts/)")
    parser.add_argument("--crops", default="real_crops",
                        help="Pasta real_crops/ (default: real_crops/)")
    parser.add_argument("--synthetic-out", default="synthetic_crops",
                        help="Pasta de saída para os crops sintéticos (default: synthetic_crops/)")
    parser.add_argument("--variations", type=int, default=3,
                        help="Variações por palavra por fonte (default: 3)")
    args = parser.parse_args()

    if not args.skip_export:
        run_step("1/3 — Export do Label Studio + crop das imagens",
                  [sys.executable, "export_and_crop.py"])
    else:
        print("A saltar o passo de export/crop (--skip-export).")

    if not args.skip_synthetic:
        fonts_dir = SCRIPT_DIR / args.fonts
        out_dir = SCRIPT_DIR / args.synthetic_out
        if not fonts_dir.exists() or not any(fonts_dir.glob("*.[tT][tT][fF]")):
            print(f"\nSem fontes .ttf em '{fonts_dir}' — a saltar geração de dados sintéticos.")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            run_step("2/3 — Geração de crops sintéticos a partir de fontes",
                      [sys.executable, "generate_words_fonts_v3.py",
                       "--fonts", args.fonts, "--crops", args.crops,
                       "--out", args.synthetic_out, "--variations", str(args.variations)])
    else:
        print("A saltar a geração de dados sintéticos (--skip-synthetic).")

    if not args.skip_train:
        run_step("3/3 — Treino do modelo (CRNN + CTC, validação cruzada)",
                  [sys.executable, "train.py"])
    else:
        print("A saltar o treino (--skip-train).")

    print("\nPipeline concluída.")


if __name__ == "__main__":
    main()