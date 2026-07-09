import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .augmentation import ImageAugmentor
from .config import TrainingConfig
from .data_utils import SplitManager
from .dataset import OCRTrainDataset, OCRValDataset, ctc_collate_fn
from .decoder import CTCDecoder
from .metrics import MetricsCalculator
from .trainer import FoldTrainer
from .vocabulary import Vocabulary


class CrossValidator:
    """
    Orquestra a validação cruzada agrupada por palavra (GroupKFold):
    prepara os dados de cada fold, treina, avalia, e no final escolhe
    e guarda o melhor modelo + resumo + curvas de loss.
    """

    def __init__(self, config: TrainingConfig, vocab: Vocabulary,
                 real_pairs: list[tuple[str, str]], synth_pairs: list[tuple[str, str]]):
        self.config = config
        self.vocab = vocab
        self.real_pairs = real_pairs
        self.synth_pairs = synth_pairs
        self.augmentor = ImageAugmentor(config.target_w, config.target_h)
        self.decoder = CTCDecoder(vocab)
        self.metrics_calc = MetricsCalculator(self.decoder, config.device)

        os.makedirs(config.output_dir, exist_ok=True)

        self.fold_val_losses: list[float] = []
        self.fold_histories: list[dict] = []
        self.fold_exact_matches: list[float] = []
        self.fold_char_accuracies: list[float] = []

    # ── Preparação de dados por fold ────────────────────────────
    def _prepare_fold_data(self, train_idx, val_idx, fold: int):
        val_words = {self.real_pairs[i][1] for i in val_idx}

        train_samples = []
        for i in train_idx:
            img_path, label = self.real_pairs[i]
            img = Image.open(img_path)
            normed = self.augmentor.normalize(img)
            for _ in range(self.config.aug_per_real + 1):
                train_samples.append((normed, label))

        synth_train = [(p, l) for p, l in self.synth_pairs if l not in val_words]
        for img_path, label in synth_train:
            img = Image.open(img_path)
            train_samples.append((self.augmentor.normalize(img), label))

        val_samples = [
            (self.augmentor.normalize(Image.open(self.real_pairs[i][0])), self.real_pairs[i][1])
            for i in val_idx
        ]

        n_real = len(train_idx)
        n_copies = n_real * (self.config.aug_per_real + 1)
        n_synth_excl = len(self.synth_pairs) - len(synth_train)
        print(f"  Treino: {len(train_samples)} "
              f"(real+cópias={n_copies}, sintético={len(synth_train)}, "
              f"sintético excluído por overlap={n_synth_excl}) | "
              f"Val: {len(val_samples)} (100% real, palavras únicas={len(val_words)})")

        fold_root = f"folds/fold_{fold + 1}"
        if os.path.exists(fold_root):
            shutil.rmtree(fold_root)
        SplitManager.save_samples(train_samples, os.path.join(fold_root, "train"))
        SplitManager.save_samples(val_samples, os.path.join(fold_root, "val"))

        return fold_root

    def _build_loaders(self, fold_root: str):
        tr_imgs, tr_lbls = SplitManager.load_split_from(fold_root, "train")
        vl_imgs, vl_lbls = SplitManager.load_split_from(fold_root, "val")

        train_ds = OCRTrainDataset(list(zip(tr_imgs, tr_lbls)), self.augmentor, self.vocab)
        val_ds = OCRValDataset(vl_imgs, vl_lbls, self.augmentor, self.vocab)

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, collate_fn=ctc_collate_fn,
            pin_memory=(self.config.device.type == "cuda"), drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.val_batch_size, shuffle=False,
            num_workers=self.config.num_workers, collate_fn=ctc_collate_fn,
            pin_memory=(self.config.device.type == "cuda"),
        )
        return train_loader, val_loader

    # ── Loop principal ───────────────────────────────────────────
    def run(self):
        groups = [label for _, label in self.real_pairs]
        kf = GroupKFold(n_splits=self.config.n_folds)

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.real_pairs, groups=groups)):
            print(f"\n{'=' * 55}")
            print(f"  FOLD {fold + 1} / {self.config.n_folds}")
            print(f"{'=' * 55}")

            fold_root = self._prepare_fold_data(train_idx, val_idx, fold)
            train_loader, val_loader = self._build_loaders(fold_root)

            trainer = FoldTrainer(self.config, self.vocab.num_classes, self.decoder, fold_index=fold)
            trainer.fit(train_loader, val_loader)

            self.fold_val_losses.append(trainer.best_val_loss)
            self.fold_histories.append(trainer.history)

            best_model = trainer.load_best_model()
            exact_rate, char_acc = self.metrics_calc.evaluate(best_model, val_loader)
            self.fold_exact_matches.append(exact_rate)
            self.fold_char_accuracies.append(char_acc)

            print(f"\nFold {fold + 1} — val_loss: {trainer.best_val_loss:.4f} | "
                  f"Exact: {exact_rate:.1f}% | Char acc: {char_acc:.1f}%")

            if self.config.quick_test_single_fold:
                break  # FIX: apenas 1 fold para teste rápido

        self._save_best_model()
        self._save_summary()
        self._save_loss_curves()

    # ── Pós-processamento ────────────────────────────────────────
    def _save_best_model(self):
        best_fold = int(np.argmin(self.fold_val_losses)) + 1
        from .model import CRNN
        best_model = CRNN(self.vocab.num_classes, self.config.target_h, self.config.target_w).to(self.config.device)
        import torch
        ckpt = os.path.join(self.config.output_dir, f"fold_{best_fold}_best.pt")
        best_model.load_state_dict(torch.load(ckpt, map_location=self.config.device))
        final_path = os.path.join(self.config.output_dir, "best_model_final.pt")
        torch.save(best_model.state_dict(), final_path)
        print(f"\nBest model → fold {best_fold} saved to {final_path}")
        self.best_fold = best_fold

    def _save_summary(self):
        summary_lines = [
            "\n" + "=" * 65,
            "Cross-Validation Results:",
            f"{'Fold':<8} {'val_loss':>10} {'Exact':>10} {'Char acc':>12} {'Fail':>10}",
            "-" * 65,
            *[
                f"  {i + 1:<6} {loss:>10.4f} {exact:>9.1f}% {char_acc:>11.1f}% {100 - exact:>9.1f}%"
                for i, (loss, exact, char_acc)
                in enumerate(zip(self.fold_val_losses, self.fold_exact_matches, self.fold_char_accuracies))
            ],
            "-" * 65,
            f"  {'Mean':<6} {np.mean(self.fold_val_losses):>10.4f} "
            f"{np.mean(self.fold_exact_matches):>9.1f}% "
            f"{np.mean(self.fold_char_accuracies):>11.1f}% "
            f"{100 - np.mean(self.fold_exact_matches):>9.1f}%",
            f"  {'Std':<6} {np.std(self.fold_val_losses):>10.4f} "
            f"{np.std(self.fold_exact_matches):>9.1f}% "
            f"{np.std(self.fold_char_accuracies):>11.1f}%",
            "=" * 65,
            f"\nBest fold: {self.best_fold} (val_loss={self.fold_val_losses[self.best_fold - 1]:.4f})",
        ]

        for line in summary_lines:
            print(line)

        summary_path = os.path.join(self.config.output_dir, "cross_validation_results.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\nResults saved → {summary_path}")

    def _save_loss_curves(self):
        n_folds_run = len(self.fold_histories)
        fig, axes = plt.subplots(1, n_folds_run, figsize=(n_folds_run * 4, 3), facecolor="#0f0e17")
        for i, hist in enumerate(self.fold_histories):
            ax = axes[i] if n_folds_run > 1 else axes
            ax.plot(hist["loss"], label="train", color="#4a90d9")
            ax.plot(hist["val_loss"], label="val", color="#e94560")
            ax.set_title(f"Fold {i + 1}", color="white")
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.legend(fontsize=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        curves_path = os.path.join(self.config.output_dir, "loss_curves.png")
        plt.savefig(curves_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Loss curves saved → {curves_path}")