"""
Teste OCR com imagem real (PyTorch)
=====================================
Carrega o best_model_final.pt, corre YOLO para detectar palavras,
e faz OCR em cada crop.

Usage:
    python test_model.py --image foto.jpg

Requirements:
    pip install torch ultralytics opencv-python pillow matplotlib
"""

import os
import cv2
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# ── Config — ajusta estes valores ─────────────────────────────
# IMPORTANTE: TARGET_W/TARGET_H têm de ser EXATAMENTE iguais aos
# usados no treino (train_ocr_torch.py), ou o modelo recebe imagens
# com uma distribuição diferente da que aprendeu.
OCR_MODEL_PATH = "results/best_model_final.pt"
VOCAB_PATH = "results/vocab.json"
YOLO_MODEL_PATH = "best_model_jul_7.pt"
TARGET_W = 256
TARGET_H = 50
YOLO_CLASSES = ["attribute", "entity"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mesmo bug de cuDNN documentado no script de treino -- ver comentário lá.
torch.backends.cudnn.enabled = False


# ══════════════════════════════════════════════════════════════
# VOCABULÁRIO — carregado do vocab.json gerado no treino.
# NÃO reconstruir a partir de real_crops/: o vocab de treino inclui
# caracteres vindos também do sintético, e reconstruir só a partir
# do real pode gerar um vocabulário mais pequeno, desalinhando os
# índices em relação ao que o modelo aprendeu.
# ══════════════════════════════════════════════════════════════
def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    characters = vocab_data["characters"]
    max_len = vocab_data["max_len"]
    # Mesma convenção do treino: índice 0 = blank do CTC, chars = 1..N
    idx_to_char = {i + 1: c for i, c in enumerate(characters)}
    return idx_to_char, len(characters), max_len


# ══════════════════════════════════════════════════════════════
# MODELO — arquitetura idêntica à de train_ocr_torch.py
# ══════════════════════════════════════════════════════════════
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, TARGET_H, TARGET_W)
            out = self.cnn(dummy)
            _, c, h, w = out.shape
            feat_dim = c * h

        self.rnn = nn.LSTM(
            input_size=feat_dim, hidden_size=256, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        self.dropout1 = nn.Dropout(0.35)
        self.rnn2 = nn.LSTM(
            input_size=512, hidden_size=256, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        self.dropout2 = nn.Dropout(0.35)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.dropout1(x)
        x, _ = self.rnn2(x)
        x = self.dropout2(x)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


def greedy_decode(log_probs: torch.Tensor, idx_to_char: dict, blank_idx: int = 0) -> list:
    pred_indices = torch.argmax(log_probs, dim=-1)
    results = []
    for seq in pred_indices:
        seq = seq.tolist()
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank_idx:
                collapsed.append(idx)
            prev = idx
        results.append("".join(idx_to_char.get(i, "") for i in collapsed))
    return results


# ══════════════════════════════════════════════════════════════
# PRÉ-PROCESSAMENTO — igual ao normalize_image do treino (resize
# preservando aspect ratio + padding em canvas branco). Usar o
# mesmo pipeline aqui é essencial: qualquer diferença (ex: CLAHE,
# threshold adaptativo) faz o modelo ver uma distribuição de
# imagem diferente da que aprendeu no treino, e degrada muito a
# qualidade das previsões mesmo com o modelo correto.
# ══════════════════════════════════════════════════════════════
def normalize_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    scale = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (TARGET_W, TARGET_H), 255)
    canvas.paste(img, ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
    return canvas


def process_image_for_ocr(cv_img) -> torch.Tensor:
    if len(cv_img.shape) == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(cv_img)
    pil_img = normalize_image(pil_img)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


# ══════════════════════════════════════════════════════════════
# YOLO HELPERS — inalterados, independentes do framework de OCR
# ══════════════════════════════════════════════════════════════
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_rotated(img, box):
    pts = np.array(box, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box_pts = order_points(cv2.boxPoints(rect))
    tl, tr, br, bl = box_pts
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width == 0 or height == 0:
        return None
    if height > width:
        width, height = height, width
        dst = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
    else:
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(box_pts, dst)
    crop = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    min_h = 30
    if crop.shape[0] < min_h:
        scale = min_h / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(image_path: str):
    from ultralytics import YOLO

    print(f"A usar dispositivo: {DEVICE}")

    # ── Vocabulário ───────────────────────────────────────────
    print(f"A carregar vocabulário: {VOCAB_PATH}")
    idx_to_char, vocab_size, max_len = load_vocab(VOCAB_PATH)
    num_classes = vocab_size + 1  # + blank
    print(f"   vocab_size={vocab_size}  num_classes={num_classes}")

    # ── Carregar modelo OCR ──────────────────────────────────
    print(f"\nA carregar modelo OCR: {OCR_MODEL_PATH}")
    ocr_model = CRNN(num_classes).to(DEVICE)
    ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE))
    ocr_model.eval()
    print("Modelo carregado.\n")

    # ── YOLO — detectar e cortar palavras ────────────────────
    print(f"YOLO a detectar palavras em: {image_path}")
    yolo = YOLO(YOLO_MODEL_PATH)
    yolo_results = yolo(image_path, imgsz=1024, conf=0.25, iou=0.45,
                         save=False, verbose=False, device="cpu")

    os.makedirs("crop_words", exist_ok=True)
    for f in os.listdir("crop_words"):
        os.remove(f"crop_words/{f}")

    cont = 0
    crop_meta = []
    result = None
    for result in yolo_results:
        img_orig = result.orig_img
        for box, cls in zip(result.obb.xyxyxyxy.tolist(), result.obb.cls.tolist()):
            class_name = result.names[int(cls)]
            if class_name not in YOLO_CLASSES:
                continue
            cont += 1
            crop = crop_rotated(img_orig, box)
            if crop is None or crop.size == 0:
                continue
            if crop.ndim == 3 and crop.shape[2] == 3:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            elif crop.ndim == 3 and crop.shape[2] == 4:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
            else:
                crop_gray = crop
            fname = f"{class_name}_{cont}.jpg"
            cv2.imwrite(f"crop_words/{fname}", crop_gray)
            crop_meta.append(fname)

    if result is not None:
        plotted = result.plot(line_width=2, labels=True, conf=True, pil=False)
        cv2.imwrite("output_yolo.jpg", plotted)
    print(f"{cont} crops guardados  |  Imagem YOLO → output_yolo.jpg\n")

    # ── OCR em cada crop ──────────────────────────────────────
    print("A correr OCR...")
    crop_files = sorted(os.listdir("crop_words"))
    results_data = []

    with torch.no_grad():
        for fname in crop_files:
            img = cv2.imread(f"crop_words/{fname}")
            if img is None or img.size == 0:
                continue
            tensor = process_image_for_ocr(img).to(DEVICE)
            log_probs = ocr_model(tensor)  # (1, T, C)
            text = greedy_decode(log_probs.cpu(), idx_to_char)[0]
            results_data.append({"file": fname, "prediction": text})
            print(f"  {fname:<35} → '{text}'")

    # ── Visualização ──────────────────────────────────────────
    n = len(results_data)
    cols = 4
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), facecolor="#0f0e17")
    axes = np.array(axes).flatten()

    i = -1
    for i, item in enumerate(results_data):
        img = cv2.imread(f"crop_words/{item['file']}")
        if img is None:
            axes[i].axis("off")
            continue
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"→ '{item['prediction']}'", fontsize=8, color="#00e676", pad=3)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle(f"OCR results — {img_name}  ({n} palavras)", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_img = f"{img_name}_ocr_results.png"
    plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nGrelha de resultados → {out_img}")
    plt.show()

    # ── Guardar resultados em ficheiro ─────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results_{img_name}_{timestamp}.json"
    txt_path = f"results_{img_name}_{timestamp}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in results_data:
            f.write(f"{item['file']} → {item['prediction']}\n")

    print(f"JSON  → {json_path}")
    print(f"TXT   → {txt_path}")
    print(f"\nTotal: {len(results_data)} palavras reconhecidas")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa OCR com uma imagem real (PyTorch).")
    parser.add_argument("--image", required=True, help="Caminho para a imagem")
    parser.add_argument("--ocr", default=OCR_MODEL_PATH,
                        help=f"Modelo OCR .pt (default: {OCR_MODEL_PATH})")
    parser.add_argument("--vocab", default=VOCAB_PATH,
                        help=f"Ficheiro vocab.json gerado no treino (default: {VOCAB_PATH})")
    parser.add_argument("--yolo", default=YOLO_MODEL_PATH,
                        help=f"Modelo YOLO (default: {YOLO_MODEL_PATH})")
    args = parser.parse_args()

    OCR_MODEL_PATH = args.ocr
    VOCAB_PATH = args.vocab
    YOLO_MODEL_PATH = args.yolo

    for p, name in [(args.image, "Imagem"), (args.ocr, "Modelo OCR"),
                     (args.yolo, "Modelo YOLO"), (args.vocab, "vocab.json")]:
        if not os.path.exists(p):
            print(f"{name} não encontrado: {p}")
            exit(1)

    run(args.image)
