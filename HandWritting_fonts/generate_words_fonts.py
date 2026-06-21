"""
Synthetic Handwriting Generator
=================================
Lê palavras do real_crops/, renderiza com fontes TTF e guarda
diretamente em real_crops/ no formato correto.

Usage:
    python generate_synthetic.py --fonts fonts/ --crops real_crops/ --variations 3

Requirements:
    pip install pillow numpy
"""

import os
import pathlib
import random
import argparse
import numpy as np
from glob import glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter

TARGET_W = 200
TARGET_H = 50


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_image(img: Image.Image) -> Image.Image:
    img   = img.convert("L")
    w, h  = img.size
    scale = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img   = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (TARGET_W, TARGET_H), 255)
    canvas.paste(img, ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
    return canvas


def render_word(word: str, font_path: str) -> Image.Image | None:
    """
    Renderiza uma palavra com a fonte dada e variações aleatórias.
    Devolve imagem PIL em escala de cinza com fundo branco.
    """
    # Tamanho aleatório da fonte — simula escrita grande/pequena
    font_size = random.randint(28, 42)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    # Medir o texto
    dummy = Image.new("L", (1, 1), 255)
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), word, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]

    if tw <= 0 or th <= 0:
        return None

    # Canvas com padding
    pad    = random.randint(6, 14)
    canvas = Image.new("L", (tw + pad * 2, th + pad * 2), 255)
    draw   = ImageDraw.Draw(canvas)

    # Cor da tinta — não só preto puro, simula caneta azul/preta com variação
    color = random.randint(0, 60)
    draw.text((pad - bbox[0], pad - bbox[1]), word,
              font=font, fill=color)

    # Rotação ligeira 3°
    if random.random() < 0.6:
        angle  = random.uniform(-3, 3)
        canvas = canvas.rotate(angle, fillcolor=255, expand=False)

    # Blur muito ligeiro
    if random.random() < 0.4:
        canvas = canvas.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.2, 0.7)))

    # Ruído ligeiro no fundo
    if random.random() < 0.5:
        arr   = np.array(canvas).astype(np.int16)
        noise = np.random.randint(-20, 20, arr.shape).astype(np.int16)  # era 0-15
        arr   = np.clip(arr + noise, 0, 255)
        canvas = Image.fromarray(arr.astype(np.uint8))

    # Brilho ligeiro
    if random.random() < 0.4:
        factor = random.uniform(0.92, 1.08)
        arr    = np.clip(np.array(canvas).astype(np.float32) * factor, 0, 255)
        arr[arr > 230] = 255
        canvas = Image.fromarray(arr.astype(np.uint8))

    # Fundo não uniforme 
    if random.random() < 0.5:
        arr = np.array(canvas).astype(np.float32)
        # Gradiente de iluminação (simula sombra de um lado)
        gradient = np.linspace(
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
            arr.shape[1]
        )
        arr = np.clip(arr * gradient[np.newaxis, :], 0, 255)
        canvas = Image.fromarray(arr.astype(np.uint8))

    # Baixo contraste — simula foto com má exposição
    if random.random() < 0.3:
        arr = np.array(canvas).astype(np.float32)
        # Comprime o range dinâmico (texto menos negro, fundo menos branco)
        arr = arr * 0.6 + random.uniform(40, 80)
        canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Bordas escuras — simula crop mal feita pelo YOLO
    if random.random() < 0.3:
        arr    = np.array(canvas).astype(np.float32)
        border = random.randint(2, 6)
        arr[:border, :]  *= random.uniform(0.3, 0.7)  # topo
        arr[-border:, :] *= random.uniform(0.3, 0.7)  # base
        arr[:, :border]  *= random.uniform(0.3, 0.7)  # esquerda
        arr[:, -border:] *= random.uniform(0.3, 0.7)  # direita
        canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        
    return normalize_image(canvas)


def load_words(crops_dir: str) -> list[str]:
    """Lê todas as labels únicas do real_crops/."""
    words = set()
    for txt in glob(os.path.join(crops_dir, "*.txt")):
        with open(txt, "r", encoding="utf-8") as f:
            w = f.read().strip()
        if w:
            words.add(w)
    return sorted(words)


def next_index(crops_dir: str) -> int:
    """Encontra o próximo índice disponível em real_crops/."""
    existing = glob(os.path.join(crops_dir, "*.png"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        try:
            indices.append(int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


# ── Main ──────────────────────────────────────────────────────────────────────
def generate(fonts_dir: str, crops_dir: str, variations: int):
    # Carregar fontes
    font_paths = sorted([
        str(p) for p in pathlib.Path(fonts_dir).rglob("*.ttf")
    ] + [
        str(p) for p in pathlib.Path(fonts_dir).rglob("*.TTF")
    ])

    if not font_paths:
        print(f"Nenhuma fonte .ttf encontrada em {fonts_dir}")
        return

    print(f"{len(font_paths)} fontes carregadas:")
    for fp in font_paths:
        print(f"   - {os.path.basename(fp)}")

    # Carregar palavras
    words = load_words(crops_dir)
    if not words:
        print(f"Nenhuma palavra encontrada em {crops_dir}")
        return

    print(f"\n{len(words)} palavras únicas encontradas")
    print(f"Total a gerar: {len(words)} × {len(font_paths)} fontes × {variations} variações"
          f" = {len(words) * len(font_paths) * variations} imagens\n")

    idx       = next_index(crops_dir)
    generated = 0
    skipped   = 0

    for word in words:
        for font_path in font_paths:
            for v in range(variations):
                img = render_word(word, font_path)
                if img is None:
                    skipped += 1
                    continue

                # Guardar
                save_name = f"{idx:05d}"
                img.save(os.path.join(crops_dir, save_name + ".png"))
                with open(os.path.join(crops_dir, save_name + ".txt"),
                          "w", encoding="utf-8") as f:
                    f.write(word)

                idx       += 1
                generated += 1

                if generated % 500 == 0:
                    print(f"  {generated} imagens geradas...", end="\r")

    print(f"\nConcluído!")
    print(f"   Geradas : {generated}")
    print(f"   Falhadas: {skipped} (caracteres não suportados pela fonte)")
    print(f"   Total em {crops_dir}: {idx} imagens")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera imagens sintéticas de escrita manual a partir de fontes TTF."
    )
    parser.add_argument("--fonts",      default="fonts",
                        help="Pasta com ficheiros .ttf (default: fonts/)")
    parser.add_argument("--crops",      default="real_crops",
                        help="Pasta real_crops/ onde guardar (default: real_crops/)")
    parser.add_argument("--variations", type=int, default=3,
                        help="Variações por palavra por fonte (default: 3)")
    args = parser.parse_args()

    for p, name in [(args.fonts, "Pasta de fontes"), (args.crops, "Pasta real_crops")]:
        if not os.path.exists(p):
            print(f"{name} não encontrada: {p}")
            exit(1)

    generate(args.fonts, args.crops, args.variations)