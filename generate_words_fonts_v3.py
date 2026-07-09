"""
Synthetic Handwriting Generator
=================================
Lê palavras do real_crops/, renderiza com fontes TTF e guarda
diretamente em real_crops/ no formato correto.

Usage:
    python generate_synthetic.py --fonts fonts/ --crops real_crops/ --out synthetic_crops/ --variations 3

Requirements:
    pip install pillow numpy
"""

import os
import pathlib
import random
import cv2
import argparse
import numpy as np
from glob import glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

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

def random_rotation(img: Image.Image) -> Image.Image:
    """Rotação ligeira """
    angle = random.uniform(-3, 3)
    arr   = np.array(img)
    h, w  = arr.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(rotated)
 
 
def ink_thickness(img: Image.Image) -> Image.Image:
    """
    Erosão ou dilatação morfológica — simula caneta mais fina ou mais grossa.
    Só afeta pixels escuros (a tinta), não o fundo.
    """
    arr    = np.array(img)
    kernel = np.ones((2, 2), np.uint8)
    if random.random() < 0.5:
        # Dilate = tinta mais grossa
        result = cv2.dilate(255 - arr, kernel, iterations=1)
    else:
        # Erode = tinta mais fina
        result = cv2.erode(255 - arr, kernel, iterations=1)
    return Image.fromarray(255 - result)

def simulate_scanner(img: Image.Image) -> Image.Image:
    """Simula imagem digitalizada — limpa, alto contraste, sem ruído."""
    arr = np.array(img).astype(np.float32)
    
    # Normaliza contraste com percentis (robusto a ruído)
    p_low  = np.percentile(arr, 5)
    p_high = np.percentile(arr, 95)
    if p_high > p_low:
        arr = np.clip((arr - p_low) / (p_high - p_low) * 255, 0, 255)
    
    # Contraste ligeiro — fator 1.2 em vez de 1.5, centrado na média real
    mean  = arr.mean()
    arr   = np.clip((arr - mean) * 1.2 + mean, 0, 255)
    
    # Ligeiro desfoque (anti-aliasing do scanner)
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    return img

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

    # Cor da tinta - não só preto puro, simula caneta azul/preta com variação
    color = random.randint(0, 60)
    draw.text((pad - bbox[0], pad - bbox[1]), word,
              font=font, fill=color)
    
    if random.random() < 0.4:
        canvas = ink_thickness(canvas)

    # Rotação ligeira 
    if random.random() < 0.6:
        canvas = random_rotation(canvas)

    # Desfoque muito ligeiro
    if random.random() < 0.4:
        canvas = canvas.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.2, 0.7)))
    
    # Simula o scanner
    if random.random() < 0.3:
        canvas = simulate_scanner(canvas)
        
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
def generate(fonts_dir: str, crops_dir: str, out_dir: str, variations: int):
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
                img.save(os.path.join(out_dir, save_name + ".png"))
                with open(os.path.join(out_dir, save_name + ".txt"),
                          "w", encoding="utf-8") as f:
                    f.write(word)

                idx       += 1
                generated += 1

                if generated % 500 == 0:
                    print(f"  {generated} imagens geradas...", end="\r")

    print(f"\nConcluído!")
    print(f"   Geradas : {generated}")
    print(f"   Falhadas: {skipped} (caracteres não suportados pela fonte)")
    print(f"   Total em {out_dir}: {idx} imagens")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera imagens sintéticas de escrita manual a partir de fontes TTF."
    )
    parser.add_argument("--fonts",      default="fonts",
                        help="Pasta com ficheiros .ttf (default: fonts/)")
    parser.add_argument("--crops",      default="real_crops",
                        help="Pasta real_crops/ onde guardar (default: real_crops/)")
    parser.add_argument("--out",        default="synthetic_crops",
                        help="Pasta onde guardar imagens sintéticas (default: synthetic_crops/)")
    parser.add_argument("--variations", type=int, default=3,
                        help="Variações por palavra por fonte (default: 3)")
    args = parser.parse_args()

    for p, name in [(args.fonts, "Pasta de fontes"), (args.crops, "Pasta real_crops"), (args.out, "Pasta de saída")]:
        if not os.path.exists(p):
            print(f"{name} não encontrada: {p}")
            exit(1)

    generate(args.fonts, args.crops, args.out, args.variations)