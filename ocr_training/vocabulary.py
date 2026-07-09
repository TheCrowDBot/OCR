import json

import torch

BLANK_IDX = 0  # Convenção PyTorch CTCLoss: índice 0 é reservado para o "blank".


class Vocabulary:
    """
    Mapeamento caracter <-> índice para o CTC.
    Os caracteres reais ocupam os índices 1..vocab_size; 0 é sempre o blank.
    """

    def __init__(self, characters: list[str], max_len: int):
        self.characters = characters
        self.max_len = max_len
        self.char_to_idx = {c: i + 1 for i, c in enumerate(characters)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(characters)}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    @property
    def num_classes(self) -> int:
        return self.vocab_size + 1  # + blank

    @classmethod
    def from_labels(cls, labels: list[str]) -> "Vocabulary":
        characters = sorted(set(c for label in labels for c in label))
        max_len = max(len(label) for label in labels)
        return cls(characters, max_len)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["characters"], data["max_len"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"characters": self.characters, "max_len": self.max_len},
                f, ensure_ascii=False, indent=2,
            )

    def encode(self, label: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)

    def decode(self, indices) -> str:
        return "".join(self.idx_to_char.get(int(i), "") for i in indices)

    def summary(self) -> str:
        return (
            f"Vocab size : {self.vocab_size}  chars: {''.join(self.characters)}\n"
            f"Max label length: {self.max_len}"
        )