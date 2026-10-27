
"""TV3 - Training and validation pipeline for CIFAR-10."""

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.tensorboard import SummaryWriter


def _to_basic_types(value: Any) -> Any:
    """Convert configuration values into serializable Python values."""
    if isinstance(value, dict):
        return {
            str(key): _to_basic_types(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_basic_types(item) for item in value]

    if isinstance(value, (Path, torch.device)):
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported configuration value: {type(value).__name__}"
    )


def _positive_integer(config: dict, key: str) -> int:
    """Validate a positive integer configuration value."""
    value = config.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"config['{key}'] must be a positive integer; got {value!r}"
        )

    return value


def _create_optimizer(
    model: nn.Module,
    config: dict,
    learning_rate: float,
    weight_decay: float,
):
    """Create an optimizer using trainable parameters only."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError(
            "No trainable parameters found. "
            "Check the model's freeze/fine-tuning strategy."
        )

    optimizer_name = str(
        config.get("optimizer") or "adam"
    ).lower()

    if optimizer_name == "adam":
        return Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. "
        "Choose adam, adamw, or sgd."
    )


def _keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    """
    Keep fully frozen BatchNorm layers in evaluation mode.

    requires_grad=False alone does not prevent BatchNorm running
    statistics from changing.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))

            if parameters and all(
                not parameter.requires_grad
                for parameter in parameters
            ):
                module.eval()


def _run_epoch(
    model: nn.Module,
    loader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer=None,
) -> dict[str, float]:
    """Run one training epoch or one validation epoch."""
    is_training = optimizer is not None

    model.train(is_training)

    if is_training:
        _keep_frozen_batchnorm_eval(model)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)

            if logits.ndim != 2 or logits.shape[1] != 10:
                raise ValueError(
                    "Expected logits with shape [batch_size, 10], "
                    f"received {tuple(logits.shape)}."
                )

            loss = loss_function(logits, labels)

            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite loss encountered.")

            if is_training:
                loss.backward()

                max_grad_norm = config_value = None
                # Gradient clipping is optional.
                # TV5 can set grad_clip_norm in the experiment config.
                del max_grad_norm, config_value

                optimizer.step()

        batch_size = labels.size(0)

        # Weight the loss by batch size so the final smaller batch
        # is handled correctly.
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int(
            (logits.detach().argmax(dim=1) == labels).sum().item()
        )
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The DataLoader contains no samples.")

    return {
        "loss": float(total_loss / total_samples),
        "accuracy": float(total_correct / total_samples),
    }


def train_model(
    model,
    dataloaders: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Train and validate a model using the shared Lab 2 contract.

    Required DataLoader keys:
        dataloaders["train"]
        dataloaders["val"]

    Required return keys:
        history
        best_checkpoint_path
        best_epoch
        best_val_accuracy
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary.")

    for key in ("train", "val"):
        if key not in dataloaders or dataloaders[key] is None:
            raise KeyError(
                f"dataloaders must contain a non-empty '{key}' loader."
            )

    epochs = _positive_integer(config, "epochs")
    batch_size = _positive_integer(config, "batch_size")

    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay") or 0.0)

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            "learning_rate must be a positive finite number."
        )

    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(
            "weight_decay must be a finite non-negative number."
        )

    device_name = str(
        config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is unavailable. "
            "Check the installed PyTorch build and GPU environment."
        )

    experiment_name = str(
        config.get("experiment_name") or "training_run"
    )

    run_dir = Path(
        config.get("run_dir")
        or Path("outputs") / "runs" / experiment_name
    )

    tensorboard_dir = Path(
        config.get("tensorboard_dir")
        or run_dir / "tensorboard"
    )

    checkpoint_dir = Path(
        config.get("checkpoint_dir")
        or run_dir / "checkpoints"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "best_model.pth"

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()

    optimizer = _create_optimizer(
        model=model,
        config=config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    saved_config = _to_basic_types(dict(config))
    saved_config["device"] = str(device)
    saved_config["batch_size"] = batch_size

    history = []
    best_epoch = None
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")

    start_time = time.perf_counter()

    # TV3 writes training/validation scalars.
    # TV4 can read these events and analyze the curves.
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    try:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                loader=dataloaders["train"],
                loss_function=loss_function,
                device=device,
                optimizer=optimizer,
            )

            val_metrics = _run_epoch(
                model=model,
                loader=dataloaders["val"],
                loss_function=loss_function,
                device=device,
            )

            train_loss = float(train_metrics["loss"])
            train_accuracy = float(train_metrics["accuracy"])
            val_loss = float(val_metrics["loss"])
            val_accuracy = float(val_metrics["accuracy"])

            metrics = (
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )

            if not all(math.isfinite(value) for value in metrics):
                raise ValueError(
                    f"Non-finite metrics encountered at epoch {epoch}."
                )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            history.append(row)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/val", val_accuracy, epoch)
            writer.flush()

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )

            # Selection criterion:
            # 1. Highest validation accuracy.
            # 2. Lowest validation loss when accuracy ties.
            # If both tie, retain the earlier checkpoint.
            is_better = (
                best_epoch is None
                or val_accuracy > best_val_accuracy
                or (
                    val_accuracy == best_val_accuracy
                    and val_loss < best_val_loss
                )
            )

            if is_better:
                best_epoch = epoch
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": int(epoch),
                    "val_accuracy": float(val_accuracy),
                    "val_loss": float(val_loss),
                    "config": saved_config,
                }

                torch.save(checkpoint, checkpoint_path)

    finally:
        # Do not leave TensorBoard's writer open after success or failure.
        writer.close()

    if best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError(
            "Training ended without saving a best checkpoint."
        )

    elapsed_seconds = float(time.perf_counter() - start_time)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training time: {elapsed_seconds:.2f} seconds")

    return {
        "history": history,
        "best_checkpoint_path": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
    }
