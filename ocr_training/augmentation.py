import io
import random

import cv2
import numpy as np
from PIL import Image, ImageFilter


class ImageAugmentor:
    """
    Agrupa a normalização de imagem e todas as operações de augmentation
    usadas no treino.
    """

    def __init__(self, target_w: int, target_h: int):
        self.target_w = target_w
        self.target_h = target_h

    # ── Normalização ────────────────────────────────────────────
    def normalize(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (self.target_w, self.target_h), 255)
        canvas.paste(img, ((self.target_w - nw) // 2, (self.target_h - nh) // 2))
        return canvas

    def resize_only(self, img: Image.Image) -> Image.Image:
        """Sem augmentation -- apenas resize, usado na validação."""
        return img.convert("L").resize((self.target_w, self.target_h), Image.BILINEAR)

    # ── Transformações individuais ──────────────────────────────
    @staticmethod
    def random_rotation(img: Image.Image) -> Image.Image:
        angle = random.uniform(-3, 3)
        arr = np.array(img)
        h, w = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return Image.fromarray(rotated)

    @staticmethod
    def ink_thickness(img: Image.Image) -> Image.Image:
        arr = np.array(img)
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            result = cv2.dilate(255 - arr, kernel, iterations=1)
        else:
            result = cv2.erode(255 - arr, kernel, iterations=1)
        return Image.fromarray(255 - result)

    @staticmethod
    def add_ruled_lines(img: Image.Image) -> Image.Image:
        arr = np.array(img).copy()
        h, w = arr.shape
        spacing = random.randint(10, 18)
        color = random.randint(200, 230)
        for y in range(0, h, spacing):
            arr[y, :] = np.minimum(arr[y, :], color)
        return Image.fromarray(arr)

    @staticmethod
    def jpeg_artifacts(img: Image.Image) -> Image.Image:
        quality = random.randint(40, 70)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    @staticmethod
    def shadow_corner(img: Image.Image) -> Image.Image:
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

    @staticmethod
    def random_perspective(img: Image.Image) -> Image.Image:
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

    @staticmethod
    def simulate_scanner(img: Image.Image) -> Image.Image:
        """Simula imagem digitalizada — limpa, alto contraste, sem ruído."""
        arr = np.array(img).astype(np.float32)

        p_low = np.percentile(arr, 5)
        p_high = np.percentile(arr, 95)
        if p_high > p_low:
            arr = np.clip((arr - p_low) / (p_high - p_low) * 255, 0, 255)

        mean = arr.mean()
        arr = np.clip((arr - mean) * 1.2 + mean, 0, 255)

        img = Image.fromarray(arr.astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        return img

    # ── Pipeline combinado ───────────────────────────────────────
    def augment(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        applied = 0

        if random.random() < 0.5:
            w, h = img.size
            m = random.uniform(-0.08, 0.08)
            img = img.transform((w, h), Image.AFFINE, (1, m, 0, 0, 1, 0), fillcolor=255)
            applied += 1

        if random.random() < 0.5:
            img = self.random_rotation(img)
            applied += 1

        if random.random() < 0.4:
            img = self.ink_thickness(img)
            applied += 1

        if random.random() < 0.4:
            radius = random.uniform(0.2, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            applied += 1

        r = random.random()
        if r < 0.25:
            img = self.jpeg_artifacts(img)
            applied += 1
        elif r < 0.50:
            img = self.simulate_scanner(img)
            applied += 1

        if random.random() < 0.4:
            factor = random.uniform(0.90, 1.10)
            arr = np.clip(np.array(img).astype(np.float32) * factor, 0, 255)
            arr[arr > 230] = 255
            img = Image.fromarray(arr.astype(np.uint8))
            applied += 1

        if random.random() < 0.20:
            img = self.shadow_corner(img)
            applied += 1

        if random.random() < 0.4:
            img = self.random_perspective(img)
            applied += 1

        if applied == 0:
            w, h = img.size
            m = random.uniform(-0.05, 0.05)
            img = img.transform((w, h), Image.AFFINE, (1, m, 0, 0, 1, 0), fillcolor=255)

        return img