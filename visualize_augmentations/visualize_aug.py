"""
Visualização de Augmentation — gera imagens de exemplo para apresentação/relatório.

Usage:
    python visualize_aug.py --image real_crops/algum_crop.png

Gera dois ficheiros:
    augmentation_individual.png  -> cada transformação isolada, com valores
                                     EXAGERADOS só para ficarem legíveis no
                                     relatório (ver nota abaixo)
    augmentation_pipeline.png    -> N passagens completas do pipeline REAL de
                                     treino, replicando exatamente o que
                                     OCRTrainDataset.__getitem__ faz:
                                         augment() -> normalize() ->
                                         add_ruled_lines ->
                                         jitter de brilho/contraste (sempre)

IMPORTANTE:
Este script importa a classe ImageAugmentor diretamente do módulo de produção
A grelha do "pipeline completo" replica o __getitem__ de OCRTrainDataset (dataset.py),
não apenas augment() -- inclui também o passo de linhas pautadas e o jitter
final de brilho/contraste que são aplicados fora do ImageAugmentor.

A grelha "individual" NÃO usa os intervalos de produção: usa valores fixos e
mais fortes só para que cada efeito seja claramente visível numa imagem de 256x50 px impressa num
relatório. Os valores reais de produção estão listados na tabela impressa na
consola e podem ser citados no relatório separadamente.
"""

import argparse
import random

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ocr_training.augmentation import ImageAugmentor

TARGET_W = 256
TARGET_H = 50

augmentor = ImageAugmentor(TARGET_W, TARGET_H)


# ── Tabela dos parâmetros reais de produção (só informativo/consola) ───────
PRODUCTION_PARAMS = [
    ("Shear inicial",        "p=0.5",  "m ~ U(-0.08, 0.08)"),
    ("Rotação",               "p=0.5",  "ângulo ~ U(-3°, 3°)"),
    ("Espessura tinta",       "p=0.4",  "dilate ou erode (50/50), kernel 2x2"),
    ("Desfoque",               "p=0.4",  "radius ~ U(0.2, 0.8)"),
    ("Artefactos JPEG",       "p=0.25", "quality ~ randint(40, 70)"),
    ("Simular scanner",       "p=0.25", "(exclusivo c/ JPEG) percentil 5-95 + contraste 1.2x"),
    ("Brilho",                 "p=0.4",  "factor ~ U(0.90, 1.10)"),
    ("Sombra de canto",       "p=0.20", "lado aleatório, border 3-8px, factor U(0.4, 0.75)"),
    ("Perspetiva",             "p=0.4",  "shift até 4px por canto"),
    ("Fallback shear",        "se nada aplicado", "m ~ U(-0.05, 0.05)"),
]

# Passos aplicados em OCRTrainDataset.__getitem__, FORA do ImageAugmentor.augment()
DATASET_EXTRA_PARAMS = [
    ("Linhas pautadas",  "p=0.3",   "após normalize(); ImageAugmentor.add_ruled_lines"),
    ("Jitter brilho",    "sempre",  "delta ~ U(-0.15, 0.15), sobre imagem já normalizada"),
    ("Jitter contraste", "sempre",  "factor ~ U(0.85, 1.15), sobre imagem já normalizada"),
]


def print_production_params():
    print("\nParâmetros REAIS usados em produção (ImageAugmentor.augment):")
    print("-" * 78)
    for name, prob, detail in PRODUCTION_PARAMS:
        print(f"  {name:<18} {prob:<18} {detail}")
    print("-" * 78)
    print("\nPassos adicionais aplicados em OCRTrainDataset.__getitem__ (dataset.py):")
    print("-" * 78)
    for name, prob, detail in DATASET_EXTRA_PARAMS:
        print(f"  {name:<18} {prob:<18} {detail}")
    print("-" * 78 + "\n")


# ── Versões "exageradas" só para a grelha individual (relatório) ───────────
# Estas funções usam a MESMA lógica/kernel dos métodos de produção, mas com
# parâmetros fixos mais fortes, para que o efeito seja visível numa imagem
# pequena impressa em papel. NÃO refletem os intervalos reais de treino.

def rotation_demo(img):
    angle = 8  # produção: U(-3, 3) — aqui fixo e maior para ficar visível
    arr = np.array(img.convert("L"))
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(rotated)


def ink_thickness_demo(img):
    arr = np.array(img.convert("L"))
    kernel = np.ones((3, 3), np.uint8)  # produção: 2x2 — aqui 3x3 para mais contraste
    result = cv2.dilate(255 - arr, kernel, iterations=1)
    return Image.fromarray(255 - result)


def blur_demo(img):
    from PIL import ImageFilter
    return img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))  # produção: U(0.2, 0.8)


def jpeg_demo(img):
    import io
    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=12)  # produção: randint(40, 70)
    buf.seek(0)
    return Image.open(buf).copy()


def brightness_demo(img):
    arr = np.array(img.convert("L")).astype(np.float32)
    arr = np.clip(arr * 1.30, 0, 255)  # produção: U(0.90, 1.10)
    arr[arr > 230] = 255
    return Image.fromarray(arr.astype(np.uint8))


def shadow_demo(img):
    arr = np.array(img.convert("L")).astype(np.float32)
    arr[:12, :] *= 0.35  # produção: border 3-8px, factor U(0.4, 0.75)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def perspective_demo(img):
    w, h = img.size
    shift = 10  # produção: até 4px
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [shift, shift], [w - shift, 0], [0, h - shift], [w - shift, h - shift],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    arr = cv2.warpPerspective(np.array(img.convert("L")), M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(arr)


def shear_demo(img):
    w, h = img.size
    m = 0.15  # produção: U(-0.08, 0.08)
    return img.convert("L").transform((w, h), Image.AFFINE, (1, m, 0, 0, 1, 0), fillcolor=255)


# ── Jitter final de brilho/contraste, igual ao OCRTrainDataset.__getitem__ ──
def brightness_contrast_jitter(img: Image.Image) -> Image.Image:
    """
    Replica exatamente o jitter aplicado em dataset.py sobre a imagem já
    normalizada (256x50), incluindo o facto de trabalhar em float [0, 1].
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    brightness_delta = random.uniform(-0.15, 0.15)
    contrast_factor = random.uniform(0.85, 1.15)
    mean = arr.mean()
    arr = (arr - mean) * contrast_factor + mean + brightness_delta
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def dataset_pipeline(img: Image.Image) -> Image.Image:
    """
    Replica OCRTrainDataset.__getitem__ (dataset.py) passo a passo:
    augment() -> normalize() -> add_ruled_lines (p=0.3) -> jitter (sempre).
    """
    out = augmentor.augment(img)
    out = augmentor.normalize(out)
    if random.random() < 0.3:
        out = augmentor.add_ruled_lines(out)
    out = brightness_contrast_jitter(out)
    return out


# ── Geração das grelhas ─────────────────────────────────────────
def make_individual_grid(img: Image.Image, out_path: str):
    base = augmentor.normalize(img)
    transforms = [
        ("Original", base),
        ("Rotação", augmentor.normalize(rotation_demo(img))),
        ("Espessura tinta", augmentor.normalize(ink_thickness_demo(img))),
        ("Desfoque", augmentor.normalize(blur_demo(img))),
        ("Artefactos JPEG", augmentor.normalize(jpeg_demo(img))),
        ("Simular scanner", augmentor.normalize(augmentor.simulate_scanner(img))),
        ("Brilho", augmentor.normalize(brightness_demo(img))),
        ("Sombra de canto", augmentor.normalize(shadow_demo(img))),
        ("Perspetiva", augmentor.normalize(perspective_demo(img))),
        ("Linhas pautadas", augmentor.add_ruled_lines(augmentor.normalize(img))),
        ("Inclinação", augmentor.normalize(shear_demo(img))),
        ("Jitter brilho/contraste", brightness_contrast_jitter(augmentor.normalize(img))),
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

    fig.suptitle(
        "Exemplos de Augmentation — Transformações",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Grelha individual guardada -> {out_path}")


def make_pipeline_grid(img: Image.Image, out_path: str, n_samples: int = 12):
    """Replica OCRTrainDataset.__getitem__ (dataset.py) passo a passo."""
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.6), facecolor="white")
    axes = axes.flatten()

    for i in range(n_samples):
        augmented = dataset_pipeline(img)
        axes[i].imshow(augmented, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Amostra {i + 1}", fontsize=9)
        axes[i].axis("off")

    for ax in axes[n_samples:]:
        ax.axis("off")

    fig.suptitle("Exemplos de Augmentation — Pipeline Completo (OCRTrainDataset)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Grelha do pipeline guardada -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera imagens de exemplo de augmentation.")
    parser.add_argument("--image", required=True, help="Caminho para um crop de exemplo (ex: real_crops/x.png)")
    parser.add_argument("--out-individual", default="augmentation_individual.png")
    parser.add_argument("--out-pipeline", default="augmentation_pipeline.png")
    parser.add_argument("--n-samples", type=int, default=12, help="Nº de amostras na grelha do pipeline")
    args = parser.parse_args()

    random.seed(42)  # reprodutibilidade da grelha do pipeline no relatório

    img = Image.open(args.image)
    print_production_params()
    make_individual_grid(img, args.out_individual)
    make_pipeline_grid(img, args.out_pipeline, args.n_samples)
