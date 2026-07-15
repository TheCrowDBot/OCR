from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    """Configuração central do pipeline de treino OCR (CRNN + CTC)."""

    real_dir: str = "real_crops"
    synth_dir: str = "synthetic_crops"
    output_dir: str = "results"
    vocab_path: str = "results/vocab.json"

    # aumentado de 200 -> seq_len=64, suficiente para max_len=29 (mínimo requerido: 59)
    target_w: int = 256
    target_h: int = 50

    aug_per_real: int = 5
    batch_size: int = 32
    val_batch_size: int = 64
    n_folds: int = 10
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    num_workers: int = 4

    early_stop_patience: int = 12
    plateau_patience: int = 8
    plateau_factor: float = 0.5
    min_lr: float = 1e-7

    # FIX: apenas 1 fold para teste rápido — muda para False para correr todos os N_FOLDS
    quick_test_single_fold: bool = False

    device: torch.device = None

    def __post_init__(self):
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NOTA: bug conhecido em builds nightly atuais do PyTorch com cuDNN 9.20.x
        # em GPUs Blackwell (sm_120) -- CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
        # em F.conv2d, reproduzido mesmo em venvs limpos (ver pytorch/pytorch#185512).
        # Desativar o cuDNN evita o crash; a GPU continua a ser usada via cuBLAS/
        # kernels CUDA nativos, só sem a aceleração específica do cuDNN.
        torch.backends.cudnn.enabled = False

    def describe_device(self) -> str:
        msg = f"A usar dispositivo: {self.device}"
        if self.device.type == "cuda":
            msg += f"\nGPU: {torch.cuda.get_device_name(0)}"
        return msg