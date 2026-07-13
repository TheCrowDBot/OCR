"""
Visualização de Augmentation — gera imagens de exemplo para apresentação/relatório.

Usage:
    python visualize_augmentations.py --image real_crops/algum_crop.png

Gera dois ficheiros:
    augmentation_individual.png  -> cada transformação isolada, com legenda
    augmentation_pipeline.png    -> N passagens completas do pipeline (como no treino)
"""

import argparse
import io
import random

import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

TARGET_W = 256
TARGET_H = 50


# ── Normalização (igual ao treino) ─────────────────────────────
def normalize(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    scale = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (TARGET_W, TARGET_H), 255)
    canvas.paste(img, ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
    return canvas


# ── Transformações individuais (idênticas ao ImageAugmentor) ──
def random_rotation(img):
    angle = random.uniform(-3, 3)
    arr = np.array(img)
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(rotated)


def ink_thickness(img):
    arr = np.array(img)
    kernel = np.ones((2, 2), np.uint8)
    result = cv2.dilate(255 - arr, kernel, iterations=1)
    return Image.fromarray(255 - result)


def add_ruled_lines(img):
    arr = np.array(img).copy()
    h, w = arr.shape
    spacing = 14
    color = 215
    for y in range(0, h, spacing):
        arr[y, :] = np.minimum(arr[y, :], color)
    return Image.fromarray(arr)


def jpeg_artifacts(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).copy()


def shadow_corner(img):
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape
    arr[:6, :] *= 0.55
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def random_perspective(img):
    w, h = img.size
    shift = 4
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.randint(0, shift), random.randint(0, shift)],
        [w - random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), h - random.randint(0, shift)],
        [w - random.randint(0, shift), h - random.randint(0, shift)],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    arr = cv2.warpPerspective(np.array(img), M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(arr)


def simulate_scanner(img):
    arr = np.array(img).astype(np.float32)
    p_low, p_high = np.percentile(arr, 5), np.percentile(arr, 95)
    if p_high > p_low:
        arr = np.clip((arr - p_low) / (p_high - p_low) * 255, 0, 255)
    mean = arr.mean()
    arr = np.clip((arr - mean) * 1.2 + mean, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(radius=0.3))


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=0.6))


def brightness(img):
    arr = np.clip(np.array(img).astype(np.float32) * 1.10, 0, 255)
    arr[arr > 230] = 255
    return Image.fromarray(arr.astype(np.uint8))


def elastic_transform(img, alpha=20, sigma=4):
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(warped.astype(np.uint8))


def full_pipeline(img):
    """Pipeline combinado, com aleatoriedade -- igual ao augment() de treino."""
    img = img.convert("L")
    if random.random() < 0.5:
        img = random_rotation(img)
    if random.random() < 0.4:
        img = ink_thickness(img)
    if random.random() < 0.4:
        img = blur(img)
    r = random.random()
    if r < 0.25:
        img = jpeg_artifacts(img)
    elif r < 0.50:
        img = simulate_scanner(img)
    if random.random() < 0.4:
        img = brightness(img)
    if random.random() < 0.20:
        img = shadow_corner(img)
    if random.random() < 0.4:
        img = random_perspective(img)
    if random.random() < 0.35:
        img = elastic_transform(img)
    return img


# ── Geração das grelhas ─────────────────────────────────────────
def make_individual_grid(img: Image.Image, out_path: str):
    base = normalize(img)
    transforms = [
        ("Original", base),
        ("Rotação", normalize(random_rotation(img))),
        ("Espessura tinta", normalize(ink_thickness(img))),
        ("Desfoque", normalize(blur(img))),
        ("Artefactos JPEG", normalize(jpeg_artifacts(img))),
        ("Simular scanner", normalize(simulate_scanner(img))),
        ("Brilho", normalize(brightness(img))),
        ("Sombra", normalize(shadow_corner(img))),
        ("Perspetiva", normalize(random_perspective(img))),
        ("Linhas pautadas", add_ruled_lines(normalize(img))),
        ("Distorção elástica", normalize(elastic_transform(img))),
    ]

    cols = 3
    rows = (len(transforms) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 1.8), facecolor="white")
    axes = axes.flatten()

    for ax, (name, t_img) in zip(axes, transforms):
        ax.imshow(t_img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    for ax in axes[len(transforms):]:
        ax.axis("off")

    fig.suptitle("Exemplos de Augmentation — Transformações Individuais", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Grelha individual guardada → {out_path}")


def make_pipeline_grid(img: Image.Image, out_path: str, n_samples: int = 12):
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.6), facecolor="white")
    axes = axes.flatten()

    for i in range(n_samples):
        augmented = normalize(full_pipeline(img))
        axes[i].imshow(augmented, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Amostra {i + 1}", fontsize=9)
        axes[i].axis("off")

    for ax in axes[n_samples:]:
        ax.axis("off")

    fig.suptitle("Exemplos de Augmentation — Pipeline Completo (usado no treino)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Grelha do pipeline guardada → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera imagens de exemplo de augmentation.")
    parser.add_argument("--image", required=True, help="Caminho para um crop de exemplo (ex: real_crops/x.png)")
    parser.add_argument("--out-individual", default="augmentation_individual.png")
    parser.add_argument("--out-pipeline", default="augmentation_pipeline.png")
    parser.add_argument("--n-samples", type=int, default=12, help="Nº de amostras na grelha do pipeline")
    args = parser.parse_args()

    img = Image.open(args.image)
    make_individual_grid(img, args.out_individual)
    make_pipeline_grid(img, args.out_pipeline, args.n_samples)