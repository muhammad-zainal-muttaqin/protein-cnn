from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Dataset

from protein_cnn.data import load_protein_arrays, split_train_val
from protein_cnn.models import build_model
from train import evaluate, train_one_epoch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["cnn1d", "cnn2d"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_all = load_protein_arrays(args.train_path)
    test = load_protein_arrays(args.test_path)
    train, val = split_train_val(train_all, val_fraction=args.val_fraction, seed=args.seed)

    train_ds = ProteinDataset(train.features, train.labels, train.mask)
    val_ds = ProteinDataset(val.features, val.labels, val.mask)
    test_ds = ProteinDataset(test.features, test.labels, test.mask)

    trial_logs_path = output_dir / "trial_logs.jsonl"
    if trial_logs_path.exists():
        trial_logs_path.unlink()

    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        width = trial.suggest_categorical("width", [96, 128, 160, 192]) if args.model == "cnn1d" else None

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        eval_batch_size = max(1, batch_size // 2)
        val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model_kwargs = {"dropout": dropout}
        if args.model == "cnn1d":
            model_kwargs["hidden_channels"] = (width, width * 2, width * 2)
        model = build_model(args.model, **model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.model)
            val_metrics = evaluate(model, val_loader, device, args.model)
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            best_val = max(best_val, val_metrics["q8_accuracy"])
            trial.report(val_metrics["q8_accuracy"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("history", history)
        trial.set_user_attr("best_val", best_val)
        with trial_logs_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "trial": trial.number,
                        "model": args.model,
                        "params": trial.params,
                        "best_val_q8_accuracy": best_val,
                        "history": history,
                    }
                )
                + "\n"
            )
        return best_val

    study = optuna.create_study(direction="maximize", study_name=f"protein_{args.model}_tuning")
    study.optimize(objective, n_trials=args.trials)

    best_params = study.best_trial.params
    batch_size = best_params["batch_size"]
    eval_batch_size = max(1, batch_size // 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_model_kwargs = {"dropout": best_params["dropout"]}
    if args.model == "cnn1d":
        best_model_kwargs["hidden_channels"] = (
            best_params["width"],
            best_params["width"] * 2,
            best_params["width"] * 2,
        )
    model = build_model(args.model, **best_model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    best_val = -1.0
    final_history = []
    best_state_path = output_dir / "best_tuned_model.pt"
    final_history_path = output_dir / "final_history.jsonl"
    if final_history_path.exists():
        final_history_path.unlink()
    for epoch in range(1, args.final_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.model)
        val_metrics = evaluate(model, val_loader, device, args.model)
        final_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        with final_history_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n"
            )
        if val_metrics["q8_accuracy"] > best_val:
            best_val = val_metrics["q8_accuracy"]
            torch.save(model.state_dict(), best_state_path)

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, args.model)

    report = {
        "model": args.model,
        "device": str(device),
        "trials": args.trials,
        "search_epochs": args.epochs,
        "final_epochs": args.final_epochs,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": best_params,
        "final_best_val_q8_accuracy": best_val,
        "final_test": test_metrics,
        "trial_summaries": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
        "final_history": final_history,
    }

    with (output_dir / "optuna_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
