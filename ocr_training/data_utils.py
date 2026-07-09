import os
from glob import glob

import numpy as np
from PIL import Image


class SampleLoader:
    """Lê pares (imagem, label) de uma pasta com ficheiros .png + .txt homónimos."""

    @staticmethod
    def load_real_samples(real_dir: str) -> list[tuple[str, str]]:
        pairs = []
        skipped = 0
        for img_path in sorted(glob(os.path.join(real_dir, "*.png"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(real_dir, base + ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                label = f.read().strip()
            if not label:
                continue
            arr = np.array(Image.open(img_path).convert("L"))
            if np.mean(arr < 200) < 0.01:
                skipped += 1
                continue
            pairs.append((img_path, label))
        print(f"Loaded {len(pairs)} samples from {real_dir}  (skipped {skipped} blank crops)")
        return pairs


class SplitManager:
    """Guarda/carrega splits de treino/validação em disco (mantido para paridade com o script original)."""

    @staticmethod
    def save_samples(samples: list[tuple[Image.Image, str]], out_dir: str) -> None:
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i, (img, label) in enumerate(samples):
            img.save(os.path.join(img_dir, f"{i:06d}.png"))
            with open(os.path.join(lbl_dir, f"{i:06d}.txt"), "w", encoding="utf-8") as f:
                f.write(label)

    @staticmethod
    def load_split_from(root: str, split: str) -> tuple[list[str], list[str]]:
        img_paths = sorted(glob(f"{root}/{split}/images/*.png"))
        lbl_list = []
        for p in img_paths:
            lp = p.replace("images", "labels").replace(".png", ".txt")
            with open(lp, "r", encoding="utf-8") as f:
                lbl_list.append(f.read().strip())
        return img_paths, lbl_list