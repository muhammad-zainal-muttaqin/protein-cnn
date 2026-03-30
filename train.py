from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from protein_cnn.data import load_protein_arrays, split_train_val, summarize_dataset
from protein_cnn.models import build_model


class ProteinDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mask: np.ndarray):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.mask[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.cross_entropy(logits, labels, reduction="none")
    loss = loss * mask.float()
    return loss.sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for features, labels, mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        if model_name == "cnn1d":
            logits = model(features.transpose(1, 2))
        else:
            logits = model(features.unsqueeze(1))

        loss = masked_loss(logits, labels, mask)
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * int(mask.sum().item())
        total_correct += int(((preds == labels) & mask).sum().item())
        total_tokens += int(mask.sum().item())

    return {
        "loss": total_loss / max(total_tokens, 1),
        "q8_accuracy": total_correct / max(total_tokens, 1),
        "tokens": total_tokens,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for features, labels, mask in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if model_name == "cnn1d":
            logits = model(features.transpose(1, 2))
        else:
            logits = model(features.unsqueeze(1))

        loss = masked_loss(logits, labels, mask)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * int(mask.sum().item())
        total_correct += int(((preds == labels) & mask).sum().item())
        total_tokens += int(mask.sum().item())

    return {
        "loss": total_loss / max(total_tokens, 1),
        "q8_accuracy": total_correct / max(total_tokens, 1),
        "tokens": total_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["cnn1d", "cnn2d"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_all = load_protein_arrays(args.train_path)
    test = load_protein_arrays(args.test_path)
    train, val = split_train_val(train_all, val_fraction=args.val_fraction, seed=args.seed)

    summaries = {
        "train_all": summarize_dataset(train_all),
        "train": summarize_dataset(train),
        "val": summarize_dataset(val),
        "test": summarize_dataset(test),
    }

    train_ds = ProteinDataset(train.features, train.labels, train.mask)
    val_ds = ProteinDataset(val.features, val.labels, val.mask)
    test_ds = ProteinDataset(test.features, test.labels, test.mask)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    eval_batch_size = max(1, args.batch_size // 2)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = -1.0
    best_state_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.model)
        val_metrics = evaluate(model, val_loader, device, args.model)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["q8_accuracy"] > best_val:
            best_val = val_metrics["q8_accuracy"]
            torch.save(model.state_dict(), best_state_path)

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, args.model)

    report = {
        "config": vars(args),
        "device": str(device),
        "dataset_summary": summaries,
        "history": history,
        "best_val_q8_accuracy": best_val,
        "test": test_metrics,
    }

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"final_test": test_metrics, "best_val_q8_accuracy": best_val}, indent=2))


if __name__ == "__main__":
    main()
