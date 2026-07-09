import torch

from .vocabulary import BLANK_IDX, Vocabulary


class CTCDecoder:
    """Greedy CTC decoding: argmax + colapso de repetições + remoção do blank."""

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def decode(self, log_probs: torch.Tensor) -> list[str]:
        """log_probs: (B, T, C) -- já em log-softmax."""
        pred_indices = torch.argmax(log_probs, dim=-1)  # (B, T)
        results = []
        for seq in pred_indices:
            seq = seq.tolist()
            collapsed = []
            prev = None
            for idx in seq:
                if idx != prev and idx != BLANK_IDX:
                    collapsed.append(idx)
                prev = idx
            results.append(self.vocab.decode(collapsed))
        return results