import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import ImageAugmentor
from .vocabulary import Vocabulary


class OCRTrainDataset(Dataset):
    """
    Recebe uma lista de (caminho da imagem, label). A imagem só é aberta
    dentro de __getitem__ (lazy loading) para não esgotar o limite de
    file descriptors do sistema quando há dezenas de milhares de amostras.
    Cada __getitem__ aplica augmentation "fresca" -- equivalente ao
    pipeline online do tf.data original (augment_tf -> process_train).
    """

    def __init__(self, samples: list[tuple[str, str]], augmentor: ImageAugmentor, vocab: Vocabulary):
        self.samples = samples
        self.augmentor = augmentor
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("L")
            img.load()

        img = self.augmentor.augment(img)
        img = self.augmentor.normalize(img)
        if random.random() < 0.3:
            img = self.augmentor.add_ruled_lines(img)

        arr = np.array(img, dtype=np.float32) / 255.0

        # brightness/contrast jitter (equivalente a tf.image.random_brightness/contrast)
        brightness_delta = random.uniform(-0.15, 0.15)
        contrast_factor = random.uniform(0.85, 1.15)
        mean = arr.mean()
        arr = (arr - mean) * contrast_factor + mean + brightness_delta
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, self.vocab.encode(label), label


class OCRValDataset(Dataset):
    """Sem augmentation -- apenas resize/normalize, igual ao process_val original."""

    def __init__(self, image_paths: list[str], labels: list[str], augmentor: ImageAugmentor, vocab: Vocabulary):
        self.image_paths = image_paths
        self.labels = labels
        self.augmentor = augmentor
        self.vocab = vocab

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = self.augmentor.resize_only(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        label = self.labels[idx]
        return tensor, self.vocab.encode(label), label


def ctc_collate_fn(batch):
    images, targets, raw_labels = zip(*batch)
    images = torch.stack(images, dim=0)  # (B, 1, H, W)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_padded = torch.zeros(len(targets), max(target_lengths).item(), dtype=torch.long)
    for i, t in enumerate(targets):
        targets_padded[i, : len(t)] = t
    return images, targets_padded, target_lengths, raw_labels