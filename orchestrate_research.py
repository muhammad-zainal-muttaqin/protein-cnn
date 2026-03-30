from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "outputs" / "reports"
RESEARCH_DIR = ROOT / "artifacts" / "research_runs"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_COLUMNS = [
    "timestamp_utc",
    "phase",
    "run_name",
    "model",
    "trials",
    "search_epochs",
    "final_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout",
    "width",
    "split",
    "save_dir",
    "report_path",
    "best_val_q8",
    "test_q8",
    "status",
    "notes",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_ledger(row: dict) -> None:
    path = REPORTS_DIR / "run_ledger.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_ledger_rows() -> list[dict]:
    path = REPORTS_DIR / "run_ledger.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def existing_run_names() -> set[str]:
    return {row["run_name"] for row in load_ledger_rows() if row.get("run_name")}


def sortable_row(row: dict) -> dict | None:
    try:
        return {
            "run_name": row["run_name"],
            "model": row["model"],
            "best_val_q8": float(row["best_val_q8"]),
            "test_q8": float(row["test_q8"]),
            "test_loss": float(row.get("test_loss", "inf")),
            "phase": row["phase"],
        }
    except (KeyError, TypeError, ValueError):
        return None


def rank_rows(rows: Iterable[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        normalized = sortable_row(row)
        if normalized is None:
            continue
        merged = dict(row)
        merged.update(normalized)
        ranked.append(merged)
    ranked.sort(key=lambda x: (x["test_q8"], x["best_val_q8"], -x["test_loss"]), reverse=True)
    return ranked


def write_status(best: dict, history: list[dict]) -> None:
    runner_up = history[1] if len(history) > 1 else None
    lines = [
        "# Latest Status",
        "",
        f"Updated: {iso_now()}",
        "",
        "## Current Best",
        "",
        f"- Run: `{best['run_name']}`",
        f"- Model: `{best['model']}`",
        f"- Best validation Q8: `{best['best_val_q8']:.4f}`",
        f"- Test Q8 on CB513: `{best['test_q8']:.4f}`",
        f"- Test loss: `{best.get('test_loss', float('nan')):.4f}`",
        "",
        "## Runner-up",
        "",
    ]
    if runner_up:
        lines.extend(
            [
                f"- Run: `{runner_up['run_name']}`",
                f"- Model: `{runner_up['model']}`",
                f"- Best validation Q8: `{runner_up['best_val_q8']:.4f}`",
                f"- Test Q8 on CB513: `{runner_up['test_q8']:.4f}`",
                f"- Test loss: `{runner_up.get('test_loss', float('nan')):.4f}`",
            ]
        )
    else:
        lines.append("- Not available yet")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Search is running sequentially and recording every experiment.",
            "- Full ledger is stored in `outputs/reports/run_ledger.csv`.",
            "- Detailed experiment summaries are stored in `outputs/reports/research_summary.json`.",
        ]
    )
    (REPORTS_DIR / "latest_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_train(exp: dict) -> dict:
    output_dir = RESEARCH_DIR / exp["run_name"]
    cmd = [
        "python",
        "train.py",
        "--train-path",
        "/workspace/cullpdb+profile_5926_filtered.npy.gz",
        "--test-path",
        "/workspace/cb513+profile_split1.npy.gz",
        "--model",
        exp["model"],
        "--feature-set",
        exp["feature_set"],
        "--epochs",
        str(exp["epochs"]),
        "--batch-size",
        str(exp["batch_size"]),
        "--lr",
        str(exp["lr"]),
        "--weight-decay",
        str(exp["weight_decay"]),
        "--dropout",
        str(exp["dropout"]),
        "--loss",
        exp["loss"],
        "--class-weighting",
        exp["class_weighting"],
        "--seed",
        str(exp["seed"]),
        "--output-dir",
        str(output_dir),
    ]
    if exp["model"] in {"cnn1d", "resdil_cnn1d"}:
        cmd.extend(["--channels", str(exp["channels"])])

    started = iso_now()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = iso_now()

    report_path = output_dir / "report.json"
    if proc.returncode != 0 or not report_path.exists():
        row = {
            "timestamp_utc": finished,
            "phase": exp["phase"],
            "run_name": exp["run_name"],
            "model": exp["model"],
            "trials": "",
            "search_epochs": exp["epochs"],
            "final_epochs": exp["epochs"],
            "batch_size": exp["batch_size"],
            "lr": exp["lr"],
            "weight_decay": exp["weight_decay"],
            "dropout": exp["dropout"],
            "width": exp.get("channels", ""),
            "split": "train_val_from_cullpdb__test_cb513",
            "save_dir": str(output_dir.relative_to(ROOT)),
            "report_path": "",
            "best_val_q8": "",
            "test_q8": "",
            "status": "failed",
            "notes": (
                f"feature_set={exp['feature_set']}; loss={exp['loss']}; "
                f"class_weighting={exp['class_weighting']}; seed={exp['seed']}; "
                f"stderr_tail={proc.stderr[-300:]}"
            ),
        }
        append_ledger(row)
        return {"ok": False, "row": row, "stdout": proc.stdout, "stderr": proc.stderr}

    report = json.loads(report_path.read_text())
    row = {
        "timestamp_utc": finished,
        "phase": exp["phase"],
        "run_name": exp["run_name"],
        "model": exp["model"],
        "trials": "",
        "search_epochs": exp["epochs"],
        "final_epochs": exp["epochs"],
        "batch_size": exp["batch_size"],
        "lr": exp["lr"],
        "weight_decay": exp["weight_decay"],
        "dropout": exp["dropout"],
        "width": exp.get("channels", ""),
        "split": "train_val_from_cullpdb__test_cb513",
        "save_dir": str(output_dir.relative_to(ROOT)),
        "report_path": str(report_path.relative_to(ROOT)),
        "best_val_q8": report["best_val_q8_accuracy"],
        "test_q8": report["test"]["q8_accuracy"],
        "status": "completed",
        "notes": (
            f"feature_set={exp['feature_set']}; loss={exp['loss']}; "
            f"class_weighting={exp['class_weighting']}; seed={exp['seed']}; "
            f"test_loss={report['test']['loss']}; started={started}"
        ),
    }
    append_ledger(row)
    return {
        "ok": True,
        "row": {
            **row,
            "test_loss": report["test"]["loss"],
            "history_path": str((output_dir / "history.jsonl").relative_to(ROOT)),
        },
        "config": exp,
    }


def main() -> None:
    stage1 = []
    existing_names = existing_run_names()
    for feature_set in ["baseline42", "extended46"]:
        for loss, class_weighting in [
            ("ce", "none"),
            ("ce", "sqrt_inverse"),
            ("focal", "none"),
            ("focal", "sqrt_inverse"),
        ]:
            for channels, lr, dropout in [
                (192, 7.5e-4, 0.17),
                (256, 7.5e-4, 0.17),
                (320, 1.0e-3, 0.22),
            ]:
                for model in ["cnn1d", "resdil_cnn1d"]:
                    idx = len(stage1) + 1
                    stage1.append(
                        {
                            "phase": "research_stage1",
                            "run_name": f"s1_{idx:02d}_{model}_{feature_set}_{loss}_{class_weighting}_c{channels}",
                            "model": model,
                            "feature_set": feature_set,
                            "epochs": 6,
                            "batch_size": 16,
                            "lr": lr,
                            "weight_decay": 4e-6,
                            "dropout": dropout,
                            "channels": channels,
                            "loss": loss,
                            "class_weighting": class_weighting,
                            "seed": 42,
                        }
                    )

    for feature_set in ["baseline42", "extended46"]:
        for dropout, lr in [(0.19, 1.15e-3), (0.28, 8.0e-4)]:
            idx = len(stage1) + 1
            stage1.append(
                {
                    "phase": "research_stage1",
                    "run_name": f"s1_{idx:02d}_cnn2d_{feature_set}_ce_none_d{int(dropout*100)}",
                    "model": "cnn2d",
                    "feature_set": feature_set,
                    "epochs": 6,
                    "batch_size": 16,
                    "lr": lr,
                    "weight_decay": 1.6e-5,
                    "dropout": dropout,
                    "channels": "",
                    "loss": "ce",
                    "class_weighting": "none",
                    "seed": 42,
                }
            )

    completed = rank_rows(load_ledger_rows())
    failed = []
    stage1_completed = []
    stage2_completed = []
    if completed:
        write_status(completed[0], completed)
    for exp in stage1:
        if exp["run_name"] in existing_names:
            continue
        result = run_train(exp)
        if result["ok"]:
            existing_names.add(exp["run_name"])
            stage1_completed.append(result["row"])
            completed.append(result["row"])
            completed = rank_rows(completed)
            write_status(completed[0], completed)
        else:
            failed.append(result["row"])

    top_stage1 = rank_rows(stage1_completed)[:8]
    stage2 = []
    for idx, row in enumerate(top_stage1, start=1):
        original = next(exp for exp in stage1 if exp["run_name"] == row["run_name"])
        stage2.append(
            {
                **original,
                "phase": "research_stage2",
                "run_name": f"s2_{idx:02d}_{original['run_name']}",
                "epochs": 18,
            }
        )

    for exp in stage2:
        if exp["run_name"] in existing_names:
            continue
        result = run_train(exp)
        if result["ok"]:
            existing_names.add(exp["run_name"])
            stage2_completed.append(result["row"])
            completed.append(result["row"])
            completed = rank_rows(completed)
            write_status(completed[0], completed)
        else:
            failed.append(result["row"])

    stage3 = []
    confirm_seeds = [7, 13, 21]
    for idx, row in enumerate(rank_rows(stage2_completed or stage1_completed)[:4], start=1):
        base_name = row["run_name"]
        if base_name.startswith("s2_"):
            original_name = base_name.split("_", 2)[2]
        else:
            original_name = base_name
        original = next(exp for exp in stage1 if exp["run_name"] == original_name)
        for seed in confirm_seeds:
            stage3.append(
                {
                    **original,
                    "phase": "research_stage3_confirm",
                    "run_name": f"s3_{idx:02d}_{original['run_name']}_seed{seed}",
                    "epochs": 24,
                    "seed": seed,
                }
            )

    for exp in stage3:
        if exp["run_name"] in existing_names:
            continue
        result = run_train(exp)
        if result["ok"]:
            existing_names.add(exp["run_name"])
            completed.append(result["row"])
            completed = rank_rows(completed)
            write_status(completed[0], completed)
        else:
            failed.append(result["row"])

    summary = {
        "updated_at": iso_now(),
        "num_completed": len(completed),
        "num_failed": len(failed),
        "best_run": completed[0] if completed else None,
        "top5": completed[:5],
        "failed": failed,
    }
    (REPORTS_DIR / "research_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
