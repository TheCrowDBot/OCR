import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .decoder import CTCDecoder
from .model import CRNN
from .vocabulary import BLANK_IDX


class FoldTrainer:
    """
    Treina um único fold: cria o modelo, corre as epochs com early stopping
    e ReduceLROnPlateau, e guarda o melhor checkpoint.
    """

    def __init__(self, config: TrainingConfig, num_classes: int, decoder: CTCDecoder, fold_index: int):
        self.config = config
        self.num_classes = num_classes
        self.decoder = decoder
        self.fold_index = fold_index  # 0-based
        self.device = config.device

        self.model = CRNN(num_classes, config.target_h, config.target_w).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=config.plateau_factor,
            patience=config.plateau_patience, min_lr=config.min_lr,
        )
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        self.checkpoint_path = os.path.join(config.output_dir, f"fold_{fold_index + 1}_best.pt")

    def _forward_loss(self, images, targets, target_lengths):
        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        log_probs = self.model(images)                       # (B, T, C)
        log_probs_t = log_probs.permute(1, 0, 2)              # (T, B, C) -- exigido pelo CTCLoss
        input_lengths = torch.full(
            (images.size(0),), log_probs.size(1), dtype=torch.long, device=self.device
        )
        loss = self.criterion(log_probs_t, targets, input_lengths, target_lengths)
        return loss, log_probs

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        for images, targets, target_lengths, _ in train_loader:
            self.optimizer.zero_grad()
            loss, _ = self._forward_loss(images, targets, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
        return running_loss / max(1, n_batches)

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        sample_preds_shown = False

        for images, targets, target_lengths, raw_labels in val_loader:
            loss, log_probs = self._forward_loss(images, targets, target_lengths)
            val_loss_sum += loss.item()
            val_batches += 1

            if not sample_preds_shown:
                decoded = self.decoder.decode(log_probs[:4].cpu())
                print("\n--- Sample predictions ---")
                for gt, pr in zip(raw_labels[:4], decoded):
                    ok = "✅" if gt == pr else "❌"
                    print(f"  {ok}  GT: '{gt}'  →  PR: '{pr}'")
                sample_preds_shown = True

        return val_loss_sum / max(1, val_batches)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Corre o treino completo do fold. Devolve o histórico de losses."""
        best_val_loss = float("inf")
        epochs_no_improve = 0
        history = {"loss": [], "val_loss": []}

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{self.config.epochs} - loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} - lr: {self.optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stop_patience:
                    print(f"Early stopping na epoch {epoch + 1} "
                          f"(sem melhoria há {self.config.early_stop_patience} epochs)")
                    break

        self.best_val_loss = best_val_loss
        self.history = history
        return history

    def load_best_model(self) -> CRNN:
        best_model = CRNN(self.num_classes, self.config.target_h, self.config.target_w).to(self.device)
        best_model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        best_model.eval()
        return best_model