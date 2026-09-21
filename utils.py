"""
Lab 3 - Utility functions for the Quantization assignment.

Covers: CIFAR-10 data loading, a CIFAR-sized MobileNetV2 (same
adaptation as Lab 2), a minimal train/eval loop, model size / latency
measurement, and helpers for dynamic and static post-training
quantization (PyTorch `torch.ao.quantization`).

The quantization *procedure itself* (fusing, qconfig, calibration loop,
convert) is left largely in the notebook, since walking through it
step-by-step is the point of the lab — this file only provides the
supporting infrastructure (data/model/train/eval/measurement) shared
with Lab 2.
"""
from __future__ import annotations

import copy
import io
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------------------------------
# Device / data  (identical setup to Lab 2, repeated here so each lab
# folder is self-contained)
# --------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    train_subset_fraction: float | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    if train_subset_fraction is not None:
        n = int(len(train_set) * train_subset_fraction)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train_set), generator=g)[:n]
        train_set = Subset(train_set, idx.tolist())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(
    num_classes: int = 10,
    pretrained: bool = True,
    cifar_stem: bool = True,
) -> nn.Module:
    """Same CIFAR-adapted MobileNetV2 as in Lab 2.

    If you completed Lab 2, you can instead load your pruned checkpoint
    from there with ``load_checkpoint`` and skip re-training here — ask
    your instructor whether that's expected for your section.
    """
    weights = (
        torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        if pretrained else None
    )
    model = torchvision.models.mobilenet_v2(weights=weights)

    if cifar_stem:
        model.features[0][0].stride = (1, 1)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Serialized size of a model's state_dict, in megabytes.

    Works for both full-precision and quantized models — quantized
    weights (int8) will show up as a smaller state_dict directly,
    unlike the pruning masks discussed in Lab 2.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 ** 2)


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location=None) -> nn.Module:
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


# --------------------------------------------------------------------------
# Train / evaluate  (identical to Lab 2)
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy in percent). Works for quantized
    models too, as long as ``device`` is CPU (quantized int8 kernels are
    CPU-only in standard PyTorch)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def fit(
    model, train_loader, test_loader, device,
    epochs: int = 3, lr: float = 1e-3, weight_decay: float = 1e-4,
    verbose: bool = True,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        if verbose:
            print(
                f"epoch {epoch}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%"
            )
    return model


@torch.no_grad()
def benchmark_latency(
    model: nn.Module, input_size=(1, 3, 32, 32), device=None,
    n_warmup: int = 10, n_runs: int = 50,
) -> float:
    """Average single-batch inference latency, in ms.

    Quantized models must be benchmarked on CPU (``device=torch.device("cpu")``)
    — PyTorch's standard int8 quantized kernels do not run on CUDA.
    """
    device = device or get_device()
    model.eval().to(device)
    dummy = torch.randn(*input_size, device=device)

    for _ in range(n_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0


# --------------------------------------------------------------------------
# Quantization helpers
# --------------------------------------------------------------------------

def dynamic_quantize(model: nn.Module, layer_types=(nn.Linear,)) -> nn.Module:
    """Post-training dynamic quantization.

    Weights of the given layer types are quantized to int8 ahead of
    time; activations are quantized on-the-fly at inference. This is
    cheapest to apply (no calibration data needed) but in a
    convolution-heavy network like MobileNetV2 (which is almost all
    `Conv2d`, not `Linear`), it only quantizes the final classifier
    layer(s) — so don't expect a large win here. It matters much more
    for architectures dominated by `Linear`/`LSTM` layers.
    """
    model = model.eval()
    return torch.ao.quantization.quantize_dynamic(
        model, qconfig_spec=set(layer_types), dtype=torch.qint8,
    )


@torch.no_grad()
def calibrate(model: nn.Module, loader, device, n_batches: int = 20) -> None:
    """Run a handful of batches through `model` in eval mode so the
    observers inserted by `torch.ao.quantization.prepare` can record
    activation statistics (min/max) needed for static quantization."""
    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(images.to(device))
