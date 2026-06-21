import os
import cv2
import random
import numpy as np
from glob import glob
import json
from label_studio_sdk import Client
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
from keras import ops
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from IPython.display import Image as IPImage, display
from sklearn.model_selection import KFold
from difflib import SequenceMatcher
import pandas as pd
import seaborn as sns

# ── Config ───────────────────────────────────────────────────
LABEL_STUDIO_URL = 'https://label.benuino.eu.org'
API_KEY = 'e1c6de79d4cc81b469902152cc5a1979936d4812'
PROJECT_ID = 7
REAL_DIR = "real_crops"
DATASET_ROOT = "final_ocr_dataset"
TARGET_W = 200 # Ajusta a largura
TARGET_H = 50 # Ajusta a altura
AUG_PER_REAL = 3 # Por cada imagem real, cria 3 versões aumentadas(rotações, distorções, ruído)
NUM_SYNTHETIC = 0 # disabled - no fonts available
batch_size = 16 # Treina 16 amostras de cada vez
#padding_token = 99 # Em palavras pequenas coloca o padding a 99, para cada label ter o mesmo comprimento 
N_FOLDS = 2 # Mudar para 5
# ── Export from Label Studio ─────────────────────────────────
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()
project = ls.get_project(PROJECT_ID)

if os.path.exists("data_yolo.zip"):
    os.unlink("data_yolo.zip")
project.export_tasks(
    export_type='YOLO_OBB_WITH_IMAGES',
    download_resources=True,
    export_location='data_yolo.zip'
)
print("YOLO export done!")

tasks = project.export_tasks(export_type='JSON')
with open("data_ocr.json", "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)
print(f"Exported {len(tasks)} tasks!")

os.system("unzip -o data_yolo.zip -d source")

# ── Crop images from Label Studio JSON ──────────────────────
with open("data_ocr.json", "r", encoding="utf-8") as f:
    tasks = json.load(f)

output_dir = "ocr_dataset/images"
os.makedirs(output_dir, exist_ok=True)
dataset = []

for task in tasks:
    image_path = task['data']['image']
    filename = image_path.split("/")[-1]
    local_image_path = f"source/images/{filename}"

    img = cv2.imread(local_image_path)
    if img is None:
        print(f"Could not read: {local_image_path}")
        continue

    img_h, img_w = img.shape[:2]
    annotations = task['annotations'][0]['result']
    boxes = {}
    texts = {}

    for ann in annotations:
        ann_id = ann['id']
        if ann['type'] == 'rectanglelabels':
            boxes[ann_id] = ann['value']
        elif ann['type'] == 'textarea':
            if ann['value']['text']:
                texts[ann_id] = ann['value']['text'][0]

    for ann_id, box in boxes.items():
        if ann_id not in texts:
            continue
        text = texts[ann_id]
        x = int((box['x'] / 100) * img_w)
        y = int((box['y'] / 100) * img_h)
        w = int((box['width'] / 100) * img_w)
        h = int((box['height'] / 100) * img_h)
        crop = img[y:y+h, x:x+w]
        if crop is None or crop.size == 0:
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        save_name = f"{task['id']}_{ann_id}.jpg"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, crop)
        dataset.append({"image": save_path, "label": text})

with open("ocr_dataset/labels.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
print(f"Total word crops saved: {len(dataset)}")

# ── Load dataset ─────────────────────────────────────────────
with open("ocr_dataset/labels.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

img_paths = [item["image"] for item in dataset]
labels = [item["label"] for item in dataset]
print(f"Total samples: {len(img_paths)}")

# ── Save real crops as PNG ────────────────────────────────────
os.makedirs(REAL_DIR, exist_ok=True)
for i, item in enumerate(dataset):
    img = Image.open(item["image"]).convert("L")
    img.save(f"{REAL_DIR}/{i:05d}.png")
    with open(f"{REAL_DIR}/{i:05d}.txt", "w", encoding="utf-8") as f:
        f.write(item["label"])
print(f"Converted {len(dataset)} samples into {REAL_DIR}/")

# ── Load real samples ─────────────────────────────────────────────────────────
def load_real_samples(real_dir=REAL_DIR):
    pairs   = []
    skipped = 0
    for img_path in sorted(glob(os.path.join(real_dir, "*.png"))):
        base     = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(real_dir, base + ".txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            label = f.read().strip()
        if not label:
            continue
        # FIX: skip blank/near-blank crops — less than 1% ink pixels
        arr = np.array(Image.open(img_path).convert("L"))
        if np.mean(arr < 200) < 0.01:
            skipped += 1
            continue
        pairs.append((img_path, label))
    print(f"Loaded {len(pairs)} real samples  (skipped {skipped} blank crops)")
    return pairs
 
real_pairs = load_real_samples()
labels_all = [p[1] for p in real_pairs]
 
# ── Vocabulary ────────────────────────────────────────────────────────────────
characters = sorted(set(c for label in labels_all for c in label))
max_len    = max(len(label) for label in labels_all)
vocab_size = len(characters)
 
print(f"Vocab size : {vocab_size}  chars: {''.join(characters)}")
print(f"Max label length: {max_len}")
 
PADDING_TOKEN = vocab_size + 1
 
char_to_num = StringLookup(vocabulary=characters, mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(),
                            mask_token=None, invert=True)
 
# ── Image normalisation ───────────────────────────────────────────────────────
def normalize_image(img: Image.Image) -> Image.Image:
    img   = img.convert("L")
    w, h  = img.size
    scale = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img   = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (TARGET_W, TARGET_H), 255)
    canvas.paste(img, ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
    return canvas
 
# ── Augmentation (PIL-based, called inside tf.data via py_function) ───────────
def random_perspective(img: Image.Image) -> Image.Image:
    """Simula ligeiro ângulo de câmara ao fotografar — shift máximo de 4px."""
    w, h  = img.size
    shift = 4   # era 5, reduzido para menos distorção
    pts1  = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2  = np.float32([
        [random.randint(0,shift), random.randint(0,shift)],
        [w-random.randint(0,shift), random.randint(0,shift)],
        [random.randint(0,shift), h-random.randint(0,shift)],
        [w-random.randint(0,shift), h-random.randint(0,shift)]
    ])
    M   = cv2.getPerspectiveTransform(pts1, pts2)
    arr = cv2.warpPerspective(np.array(img), M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(arr)
 
 
def random_rotation(img: Image.Image) -> Image.Image:
    """Rotação ligeira ±3° — simula crop YOLO ligeiramente torto."""
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
    Só afecta pixels escuros (a tinta), não o fundo.
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
 
 
def add_ruled_lines(img: Image.Image) -> Image.Image:
    """
    Adiciona linhas horizontais subtis — simula papel pautado
    (comum nas tuas imagens de dataset).
    """
    arr    = np.array(img).copy()
    h, w   = arr.shape
    # Espaçamento aleatório entre 10-18px, cor cinza claro (200-230)
    spacing = random.randint(10, 18)
    color   = random.randint(200, 230)
    for y in range(0, h, spacing):
        arr[y, :] = np.minimum(arr[y, :], color)
    return Image.fromarray(arr)
 
 
def jpeg_artifacts(img: Image.Image) -> Image.Image:
    """
    Simula compressão JPEG de foto de telemóvel — artefactos à volta dos traços.
    Usa qualidade baixa (40-70) para criar ruído realista.
    """
    import io
    quality = random.randint(40, 70)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def shadow_corner(img: Image.Image) -> Image.Image:
    """
    Escurece um canto/borda aleatória — simula sombra ao fotografar
    uma folha de papel com luz lateral (visto nas crops reais).
    """
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape
    side = random.choice(["top", "bottom", "left", "right"])
    border = random.randint(3, 8)
    factor = random.uniform(0.4, 0.75)
    if side == "top":
        arr[:border, :] *= factor
    elif side == "bottom":
        arr[-border:, :] *= factor
    elif side == "left":
        arr[:, :border] *= factor
    else:
        arr[:, -border:] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
 
 
def augment_pil(img: Image.Image) -> Image.Image:
    """
    Pipeline de augmentation conservador para OCR de escrita manual
    em fundo branco. Cada transformação é opcional e calibrada para
    não destruir a legibilidade.
    """
    img     = img.convert("L")
    applied = 0
 
    # 1. Slant ligeiro — simula inclinação da escrita (reduzido de ±0.15 para ±0.08)
    if random.random() < 0.5:
        w, h = img.size
        m    = random.uniform(-0.08, 0.08)
        img  = img.transform((w, h), Image.AFFINE, (1, m, 0, 0, 1, 0),
                              fillcolor=255)
        applied += 1
 
    # 2. Rotação ligeira — crop YOLO raramente sai perfeitamente horizontal
    if random.random() < 0.5:
        img = random_rotation(img)
        applied += 1
 
    # 3. Perspectiva subtil — ângulo de câmara
    if random.random() < 0.4:
        img = random_perspective(img)
        applied += 1
 
    # 4. Espessura da tinta — caneta diferente entre pessoas
    if random.random() < 0.4:
        img = ink_thickness(img)
        applied += 1
 
    # 5. Blur muito ligeiro — foco de câmara (radius máx 0.8, era 1.2)
    if random.random() < 0.4:
        radius = random.uniform(0.2, 0.8)
        img    = img.filter(ImageFilter.GaussianBlur(radius=radius))
        applied += 1
 
    # 6. Linhas do papel — presente em muitas imagens do teu dataset
    if random.random() < 0.3:
        img = add_ruled_lines(img)
        applied += 1
 
    # 7. Artefactos JPEG — fotos de telemóvel têm sempre compressão
    if random.random() < 0.4:
        img = jpeg_artifacts(img)
        applied += 1
 
    # 8. Brilho ligeiro — iluminação variável (factor ±10%, era ±15%)
    if random.random() < 0.4:
        factor = random.uniform(0.90, 1.10)
        arr    = np.clip(np.array(img).astype(np.float32) * factor, 0, 255)
        # Garante que o fundo não fica cinzento — força pixels muito claros a branco
        arr[arr > 230] = 255
        img = Image.fromarray(arr.astype(np.uint8))
        applied += 1
    
    # 9. Sombra de canto/borda — simula foto com luz lateral
    if random.random() < 0.25:
        img = shadow_corner(img)
        applied += 1
 
    # Garante que pelo menos uma augmentation disparou
    if applied == 0:
        w, h = img.size
        m    = random.uniform(-0.05, 0.05)
        img  = img.transform((w, h), Image.AFFINE, (1, m, 0, 0, 1, 0),
                              fillcolor=255)
 
    return img
 
# ── Save fold splits ──────────────────────────────────────────────────────────
def save_samples(samples, out_dir):
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i, (img, label) in enumerate(samples):
        img.save(os.path.join(img_dir, f"{i:06d}.png"))
        with open(os.path.join(lbl_dir, f"{i:06d}.txt"), "w", encoding="utf-8") as f:
            f.write(label)
 
def load_split_from(root, split):
    img_paths = sorted(glob(f"{root}/{split}/images/*.png"))
    lbl_list  = []
    for p in img_paths:
        lp = p.replace("images", "labels").replace(".png", ".txt")
        with open(lp, "r", encoding="utf-8") as f:
            lbl_list.append(f.read().strip())
    return img_paths, lbl_list
 
# ── tf.data preprocessing ─────────────────────────────────────────────────────
def read_image_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [TARGET_H, TARGET_W])
    img = tf.cast(img, tf.float32) / 255.0
    return img
 
def vectorize_label(label):
    chars  = tf.strings.unicode_split(label, input_encoding="UTF-8")
    tokens = char_to_num(chars)
    pad    = max_len - tf.shape(tokens)[0]
    tokens = tf.pad(tokens, [[0, pad]], constant_values=PADDING_TOKEN)
    return tokens
 
# FIX: augmentation applied on-the-fly via py_function so every epoch
# sees different distortions instead of the same baked-in augmented files.
def augment_tf(image_path, label):
    def _aug(path):
        img = Image.open(path.numpy().decode()).convert("L")
        img = augment_pil(img)
        img = normalize_image(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.reshape(TARGET_H, TARGET_W, 1)
    img = tf.py_function(_aug, [image_path], tf.float32)
    img.set_shape([TARGET_H, TARGET_W, 1])
    return img
 
def process_train(image_path, label):
    image = augment_tf(image_path, label)
    # Extra TF-native augmentation on top
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return {"image": image, "label": vectorize_label(label)}
 
def process_val(image_path, label):
    image = read_image_tf(image_path)
    return {"image": image, "label": vectorize_label(label)}
 
def prepare_dataset(image_paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(2048)
        ds = ds.map(process_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(process_val, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ── CTC Layer ─────────────────────────────────────────────────────────────────
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
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
 
    def get_config(self):
        return super().get_config()

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    input_img = keras.Input(shape=(TARGET_H, TARGET_W, 1), name="image")
    labels    = keras.Input(name="label", shape=(None,))
 
    # CNN backbone
    x = keras.layers.Conv2D(64,  3, padding="same", activation="relu")(input_img)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 1))(x)   # only halve height
    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 1))(x)   # only halve height

    shape    = x.shape                      # (None, H', W', C)
    seq_len  = shape[2]                     # width dimension = time steps
    feat_dim = shape[1] * shape[3]          # height * channels = features per step
    x = keras.layers.Reshape((seq_len, feat_dim))(x)
 
    # Bidirectional LSTM
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True,
                                                      dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True,
                                                      dropout=0.25))(x)
 
    # Output — +1 for CTC blank token (index 0 used by StringLookup OOV slot)
    x = keras.layers.Dense(vocab_size + 2, activation="softmax", name="logits")(x)
 
    output = CTCLayer()(labels, x)
 
    model = keras.Model(inputs=[input_img, labels], outputs=output)
 
    opt = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=5.0)
    model.compile(optimizer=opt)
    return model
 
# ── Decode helpers ────────────────────────────────────────────────────────────
def decode_label(label):
    label = tf.cast(label, tf.int32)
    label = tf.boolean_mask(label, tf.not_equal(label, PADDING_TOKEN))
    return tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
 
def decode_batch_predictions(preds):
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    results   = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
    out = []
    for res in results:
        res  = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        out.append(text)
    return out
 
class DisplayPredictions(keras.callbacks.Callback):
    def __init__(self, dataset, pred_model, num_samples=4):
        self.dataset     = dataset.take(1)
        self.num_samples = num_samples
        self.pred_model  = pred_model
 
    def on_epoch_end(self, epoch, logs=None):
        for batch in self.dataset:
            imgs   = batch["image"][:self.num_samples]
            lbls   = batch["label"][:self.num_samples]
            preds  = self.pred_model.predict(imgs, verbose=0)
            decoded = decode_batch_predictions(preds)
            print("\n--- Sample predictions ---")
            for i in range(min(self.num_samples, len(decoded))):
                gt = decode_label(lbls[i])
                pr = decoded[i]
                ok = "✅" if gt == pr else "❌"
                print(f"  {ok}  GT: '{gt}'  →  PR: '{pr}'")
            break
 
# ── K-Fold Cross Validation ───────────────────────────────────────────────────
os.makedirs("ocr_receipt", exist_ok=True)
 
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
 
fold_val_losses     = []
fold_histories      = []
fold_exact_matches  = []
fold_char_accuracies = []
 
for fold, (train_idx, val_idx) in enumerate(kf.split(real_pairs)):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold+1} / {N_FOLDS}")
    print(f"{'='*55}")
 
    # Build training set: real + augmented copies saved to disk once per fold
    train_samples = []
    for i in train_idx:
        img_path, label = real_pairs[i]
        img = Image.open(img_path)
        train_samples.append((normalize_image(img), label))
        for _ in range(AUG_PER_REAL):
            train_samples.append((normalize_image(augment_pil(img)), label))
 
    val_samples = [
        (normalize_image(Image.open(real_pairs[i][0])), real_pairs[i][1])
        for i in val_idx
    ]
 
    fold_root = f"folds/fold_{fold+1}"
    save_samples(train_samples, os.path.join(fold_root, "train"))
    save_samples(val_samples,   os.path.join(fold_root, "val"))
 
    tr_imgs, tr_lbls = load_split_from(fold_root, "train")
    vl_imgs, vl_lbls = load_split_from(fold_root, "val")
 
    train_ds = prepare_dataset(tr_imgs, tr_lbls, training=True)
    val_ds   = prepare_dataset(vl_imgs, vl_lbls, training=False)
 
    keras.backend.clear_session()
    model = build_model()
 
    pred_model = keras.models.Model(
        inputs=model.input[0],
        outputs=model.get_layer(name="logits").output
    )
 
    ckpt_path = f"ocr_receipt/fold_{fold+1}_best.keras"
 
    # cosine decay gives a much smoother loss curve than ReduceLROnPlateau alone
    total_steps  = len(tr_imgs) // batch_size * 150
    warmup_steps = total_steps // 10
    cosine_decay = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=total_steps - warmup_steps,
        t_mul=1.0, m_mul=0.9, alpha=1e-6
    )
 
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, monitor="val_loss"
        ),
        # Keep ReduceLROnPlateau as a safety net for plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8,
            min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25,
            restore_best_weights=True, verbose=1
        ),
        DisplayPredictions(val_ds, pred_model),
    ]
 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=175,
        callbacks=callbacks,
        verbose=1
    )
 
    best_val_loss = min(history.history["val_loss"])
    fold_val_losses.append(best_val_loss)
    fold_histories.append(history.history)
 
    # ── Per-fold metrics ──────────────────────────────────────────────────────
    best_model = keras.models.load_model(
        ckpt_path, custom_objects={"CTCLayer": CTCLayer}
    )
    best_pred = keras.models.Model(
        inputs=best_model.input[0],
        outputs=best_model.get_layer(name="logits").output
    )
 
    exact = 0
    total_chars   = 0
    correct_chars = 0
 
    for img_path, true_label in zip(vl_imgs, vl_lbls):
        img  = tf.expand_dims(
            tf.io.decode_png(tf.io.read_file(img_path), channels=1), 0
        )
        img  = tf.image.resize(img, [TARGET_H, TARGET_W])
        img  = tf.cast(img, tf.float32) / 255.0
        pred = best_pred.predict(img, verbose=0)
        pred_label = decode_batch_predictions(pred)[0]
 
        if pred_label == true_label:
            exact += 1
 
        matcher = SequenceMatcher(None, true_label, pred_label)
        correct_chars += sum(i2-i1 for tag,i1,i2,j1,j2 in matcher.get_opcodes()
                             if tag == "equal")
        total_chars += len(true_label)
 
    exact_rate   = exact / len(vl_lbls) * 100
    char_acc     = correct_chars / total_chars * 100 if total_chars > 0 else 0
    fold_exact_matches.append(exact_rate)
    fold_char_accuracies.append(char_acc)
 
    print(f"\nFold {fold+1} — val_loss: {best_val_loss:.4f} | "
          f"Exact: {exact_rate:.1f}% | Char acc: {char_acc:.1f}%")
    
    break
 
# ── Select and save best model ────────────────────────────────────────────────
best_fold = int(np.argmin(fold_val_losses)) + 1
best_model = keras.models.load_model(
    f"ocr_receipt/fold_{best_fold}_best.keras",
    custom_objects={"CTCLayer": CTCLayer}
)
best_model.save("ocr_receipt/best_model_final.keras")
print(f"\nBest model → fold {best_fold} saved to ocr_receipt/best_model_final.keras")
 
# ── Cross-validation summary ──────────────────────────────────────────────────
summary_lines = [
    "\n" + "="*65,
    "Cross-Validation Results:",
    f"{'Fold':<8} {'val_loss':>10} {'Exact':>10} {'Char acc':>12} {'Fail':>10}",
    "-"*65,
    *[
        f"  {i+1:<6} {loss:>10.4f} {exact:>9.1f}% {char_acc:>11.1f}% {100-exact:>9.1f}%"
        for i, (loss, exact, char_acc)
        in enumerate(zip(fold_val_losses, fold_exact_matches, fold_char_accuracies))
    ],
    "-"*65,
    f"  {'Mean':<6} {np.mean(fold_val_losses):>10.4f} "
    f"{np.mean(fold_exact_matches):>9.1f}% "
    f"{np.mean(fold_char_accuracies):>11.1f}% "
    f"{100 - np.mean(fold_exact_matches):>9.1f}%",
    f"  {'Std':<6} {np.std(fold_val_losses):>10.4f} "
    f"{np.std(fold_exact_matches):>9.1f}% "
    f"{np.std(fold_char_accuracies):>11.1f}%",
    "="*65,
    f"\nBest fold: {best_fold} (val_loss={fold_val_losses[best_fold-1]:.4f})",
]

for line in summary_lines:
    print(line)

with open("cross_validation_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nResults saved → cross_validation_results.txt")
 
# ── Loss curves ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_FOLDS, figsize=(N_FOLDS * 4, 3), facecolor='#0f0e17')
for i, hist in enumerate(fold_histories):
    ax = axes[i] if N_FOLDS > 1 else axes
    ax.plot(hist["loss"],     label="train", color="#4a90d9")
    ax.plot(hist["val_loss"], label="val",   color="#e94560")
    ax.set_title(f"Fold {i+1}", color="white")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.legend(fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
plt.tight_layout()
plt.savefig("ocr_receipt/loss_curves.png", dpi=150,
            facecolor=fig.get_facecolor())
print("Loss curves saved → ocr_receipt/loss_curves.png")