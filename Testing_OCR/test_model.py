"""
Teste OCR com imagem real
==========================
Carrega o best_model_final.keras, corre YOLO para detectar palavras,
e faz OCR em cada crop.

Usage:
    python test_ocr.py --image foto.jpg

Requirements:
    pip install tensorflow ultralytics opencv-python pillow matplotlib
"""

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
from keras import ops
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from glob import glob

# ── Config — ajusta estes valores ─────────────────────────────────────────────
OCR_MODEL_PATH  = "ocr_receipt/best_model_final.keras"
YOLO_MODEL_PATH = "best_april_10_2026_at_14_56.pt"
REAL_DIR        = "real_crops"
TARGET_W        = 200
TARGET_H        = 50
YOLO_CLASSES    = ["attribute", "entity"]

# ══════════════════════════════════════════════════════════════════════════════
# VOCABULÁRIO — reconstruído a partir do real_crops/ (igual ao treino)
# ══════════════════════════════════════════════════════════════════════════════
def build_vocab(real_dir):
    labels = []
    for txt in sorted(glob(os.path.join(real_dir, "*.txt"))):
        with open(txt, "r", encoding="utf-8") as f:
            label = f.read().strip()
        if label:
            labels.append(label)
    characters    = sorted(set(c for label in labels for c in label))
    max_len       = max(len(l) for l in labels)
    vocab_size    = len(characters)
    PADDING_TOKEN = vocab_size + 1
    char_to_num   = StringLookup(vocabulary=characters, mask_token=None)
    num_to_char   = StringLookup(vocabulary=char_to_num.get_vocabulary(),
                                  mask_token=None, invert=True)
    return num_to_char, PADDING_TOKEN, vocab_size

# ══════════════════════════════════════════════════════════════════════════════
# CTC LAYER — necessário para carregar o modelo
# ══════════════════════════════════════════════════════════════════════════════
@keras.utils.register_keras_serializable()
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len    = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")
        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        self.add_loss(self.loss_fn(y_true, y_pred, input_length, label_length))
        return y_pred

    def get_config(self):
        return super().get_config()

# ══════════════════════════════════════════════════════════════════════════════
# DECODE
# ══════════════════════════════════════════════════════════════════════════════
def decode_batch_predictions(preds, num_to_char):
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    results   = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
    out = []
    for res in results:
        res  = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        out.append(text)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# YOLO HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def order_points(pts):
    rect     = np.zeros((4, 2), dtype="float32")
    s        = pts.sum(axis=1)
    rect[0]  = pts[np.argmin(s)];  rect[2] = pts[np.argmax(s)]
    diff     = np.diff(pts, axis=1)
    rect[1]  = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def crop_rotated(img, box):
    pts      = np.array(box, dtype=np.float32)
    rect     = cv2.minAreaRect(pts)
    box_pts  = order_points(cv2.boxPoints(rect))
    tl, tr, br, bl = box_pts
    width    = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    height   = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    if width == 0 or height == 0:
        return None
    if height > width:
        width, height = height, width
        dst = np.array([[0,height-1],[0,0],[width-1,0],[width-1,height-1]], dtype="float32")
    else:
        dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
    
    M    = cv2.getPerspectiveTransform(box_pts, dst)
    crop = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # <- melhor interpolação e evita bordas pretas
    
    # Upscale se a crop for muito pequena antes de passar ao OCR
    min_h = 30
    if crop.shape[0] < min_h:
        scale = min_h / crop.shape[0]
        crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return crop

def process_image_for_ocr(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Redimensionar com interpolação melhor
    image = cv2.resize(image, (TARGET_W, TARGET_H), 
                       interpolation=cv2.INTER_CUBIC)
    
    # Melhorar contraste com CLAHE (melhor que equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))
    image = clahe.apply(image)
    
    # Remover ruído suave
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Binarização adaptativa para lidar com iluminação irregular
    image = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=10
    )
    
    # Normalizar
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(tf.expand_dims(image, -1), 0)
    
    return image

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run(image_path: str):
    from ultralytics import YOLO

    # ── Vocabulário ───────────────────────────────────────────────────────────
    print("A reconstruir vocabulário a partir do real_crops/...")
    num_to_char, PADDING_TOKEN, vocab_size = build_vocab(REAL_DIR)
    print(f"   vocab_size={vocab_size}  PADDING_TOKEN={PADDING_TOKEN}")

    # ── Carregar modelo OCR ───────────────────────────────────────────────────
    print(f"\nA carregar modelo OCR: {OCR_MODEL_PATH}")
    ocr_model        = keras.models.load_model(OCR_MODEL_PATH,
                                                custom_objects={"CTCLayer": CTCLayer})
    prediction_model = keras.models.Model(
        inputs=ocr_model.input[0],
        outputs=ocr_model.get_layer(name="logits").output
    )
    print("Modelo carregado.\n")

    # ── YOLO — detectar e cortar palavras ─────────────────────────────────────
    print(f"YOLO a detectar palavras em: {image_path}")
    yolo       = YOLO(YOLO_MODEL_PATH)
    yolo_results = yolo(image_path, imgsz=1024, conf=0.25, iou=0.45,
                        save=False, verbose=False, device="cpu")

    os.makedirs("crop_words", exist_ok=True)
    for f in os.listdir("crop_words"):
        os.remove(f"crop_words/{f}")

    cont = 0
    crop_meta = []   # [(fname, class_name, box_coords)]
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
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            fname     = f"{class_name}_{cont}.jpg"
            cv2.imwrite(f"crop_words/{fname}", crop_gray)
            crop_meta.append(fname)

    # Guardar imagem com boxes desenhadas
    plotted = result.plot(line_width=2, labels=True, conf=True, pil=False)
    cv2.imwrite("output_yolo.jpg", plotted)
    print(f"{cont} crops guardados  |  Imagem YOLO → output_yolo.jpg\n")

    # ── OCR em cada crop ──────────────────────────────────────────────────────
    print("🔤 A correr OCR...")
    crop_files   = sorted(os.listdir("crop_words"))
    results_data = []

    for fname in crop_files:
        img = cv2.imread(f"crop_words/{fname}")
        if img is None or img.size == 0:
            continue
        processed  = process_image_for_ocr(img)
        pred       = prediction_model.predict(processed, verbose=0)
        text       = decode_batch_predictions(pred, num_to_char)[0]
        results_data.append({"file": fname, "prediction": text})
        print(f"  {fname:<35} → '{text}'")

    # ── Visualização ──────────────────────────────────────────────────────────
    n     = len(results_data)
    cols  = 4
    rows  = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5),
                              facecolor='#0f0e17')
    axes = np.array(axes).flatten()

    for i, item in enumerate(results_data):
        img = cv2.imread(f"crop_words/{item['file']}")
        if img is None:
            axes[i].axis("off")
            continue
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"→ '{item['prediction']}'", fontsize=8,
                           color='#00e676', pad=3)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle(f"OCR results — {img_name}  ({n} palavras)",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_img = f"{img_name}_ocr_results.png"
    plt.savefig(out_img, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nGrelha de resultados → {out_img}")
    plt.show()

    # ── Guardar resultados em ficheiro ────────────────────────────────────────
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results_{img_name}_{timestamp}.json"
    txt_path  = f"results_{img_name}_{timestamp}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in results_data:
            f.write(f"{item['file']} → {item['prediction']}\n")

    print(f"JSON  → {json_path}")
    print(f"TXT   → {txt_path}")
    print(f"\nTotal: {len(results_data)} palavras reconhecidas")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa OCR com uma imagem real.")
    parser.add_argument("--image",    required=True, help="Caminho para a imagem")
    parser.add_argument("--ocr",      default=OCR_MODEL_PATH,
                        help=f"Modelo OCR (default: {OCR_MODEL_PATH})")
    parser.add_argument("--yolo",     default=YOLO_MODEL_PATH,
                        help=f"Modelo YOLO (default: {YOLO_MODEL_PATH})")
    parser.add_argument("--crops",    default=REAL_DIR,
                        help=f"Pasta real_crops para vocab (default: {REAL_DIR})")
    args = parser.parse_args()

    OCR_MODEL_PATH  = args.ocr
    YOLO_MODEL_PATH = args.yolo
    REAL_DIR        = args.crops

    for p, name in [(args.image, "Imagem"), (args.ocr, "Modelo OCR"),
                    (args.yolo, "Modelo YOLO"), (args.crops, "real_crops")]:
        if not os.path.exists(p):
            print(f"{name} não encontrado: {p}")
            exit(1)

    run(args.image)