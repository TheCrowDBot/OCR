import os
import cv2
import json
import numpy as np
from glob import glob
from label_studio_sdk import Client
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
# ── Config ───────────────────────────────────────────────────
LABEL_STUDIO_URL = 'https://label.benuino.eu.org'
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
PROJECT_ID = 7
REAL_DIR = "real_crops"
MIN_ASPECT_RATIO = 1.0  # crops com width/height abaixo disto são anotações degeneradas 
#(caixa desenhada demasiado estreita/alta no Label Studio) e não têm texto legível -- não são gravados
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
skipped_degenerate = 0

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

        crop_h, crop_w = crop.shape[:2]
        if crop_h == 0 or crop_w / crop_h < MIN_ASPECT_RATIO:
            skipped_degenerate += 1
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        save_name = f"{task['id']}_{ann_id}.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, crop)
        dataset.append({"image": save_path, "label": text})

print(f"Crops degenerados descartados (aspect ratio < {MIN_ASPECT_RATIO}): {skipped_degenerate}")

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
