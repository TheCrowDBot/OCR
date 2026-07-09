import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """CNN + BiLSTM (x2) + CTC head."""

    def __init__(self, num_classes: int, target_h: int, target_w: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # só reduz altura
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),  # só reduz altura
        )

        # Calcula dinamicamente feat_dim (canais * altura restante) com um forward dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 1, target_h, target_w)
            out = self.cnn(dummy)
            _, c, h, w = out.shape
            self.seq_len = w
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
        # x: (B, 1, H, W)
        x = self.cnn(x)                      # (B, C, H', W')
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2)             # (B, W', C, H')
        x = x.reshape(b, w, c * h)            # (B, seq_len, feat_dim)

        x, _ = self.rnn(x)
        x = self.dropout1(x)
        x, _ = self.rnn2(x)
        x = self.dropout2(x)

        logits = self.fc(x)                  # (B, seq_len, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs  # (B, T, C) -- convertido para (T, B, C) antes do CTCLoss

    def check_seq_len(self, max_len: int) -> None:
        """Aviso defensivo: seq_len tem de ser suficiente para o CTC alinhar max_len."""
        min_required = 2 * max_len + 1
        print(f"seq_len do modelo: {self.seq_len} | mínimo necessário p/ max_len={max_len}: {min_required}")
        if self.seq_len < min_required:
            print(
                "AVISO: seq_len insuficiente para o CTC alinhar as labels mais longas. "
                "Isto vai gerar losses muito altas para essas amostras (outliers). "
                "Considera reduzir menos a largura no CNN, aumentar TARGET_W, "
                "ou rever/filtrar labels anormalmente longas no dataset."
            )