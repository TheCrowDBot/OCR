from difflib import SequenceMatcher

import torch
from torch.utils.data import DataLoader

from .decoder import CTCDecoder
from .model import CRNN


class MetricsCalculator:
    """Calcula exact-match rate e char accuracy de um modelo sobre um DataLoader de validação."""

    def __init__(self, decoder: CTCDecoder, device: torch.device):
        self.decoder = decoder
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: CRNN, val_loader: DataLoader) -> tuple[float, float]:
        model.eval()
        exact = 0
        total_chars = 0
        correct_chars = 0

        for images, _targets, _target_lengths, raw_labels in val_loader:
            images = images.to(self.device)
            log_probs = model(images)
            decoded = self.decoder.decode(log_probs.cpu())
            for true_label, pred_label in zip(raw_labels, decoded):
                if pred_label == true_label:
                    exact += 1
                matcher = SequenceMatcher(None, true_label, pred_label)
                correct_chars += sum(
                    i2 - i1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == "equal"
                )
                total_chars += len(true_label)

        exact_rate = exact / len(val_loader.dataset) * 100
        char_acc = correct_chars / total_chars * 100 if total_chars > 0 else 0
        return exact_rate, char_acc