from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


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


def load_ledger_rows() -> list[dict]:
    path = REPORTS_DIR / "run_ledger.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_ledger(row: dict) -> None:
    path = REPORTS_DIR / "run_ledger.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def existing_run_names() -> set[str]:
    return {row["run_name"] for row in load_ledger_rows() if row.get("run_name")}


def extract_test_loss(notes: str) -> float:
    match = re.search(r"test_loss=([0-9.]+)", notes or "")
    return float(match.group(1)) if match else float("inf")


def rank_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        try:
            normalized.append(
                {
                    **row,
                    "_best_val": float(row["best_val_q8"]),
                    "_test_q8": float(row["test_q8"]),
                    "_test_loss": extract_test_loss(row.get("notes", "")),
                }
            )
        except Exception:
            continue
    normalized.sort(key=lambda x: (x["_test_q8"], x["_best_val"], -x["_test_loss"]), reverse=True)
    return normalized


def write_status() -> None:
    ranked = rank_rows(load_ledger_rows())
    if not ranked:
        return
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    lines = [
        "# Latest Status",
        "",
        f"Updated: {iso_now()}",
        "",
        "## Current Best",
        "",
        f"- Run: `{best['run_name']}`",
        f"- Model: `{best['model']}`",
        f"- Best validation Q8: `{best['_best_val']:.4f}`",
        f"- Test Q8 on CB513: `{best['_test_q8']:.4f}`",
        f"- Test loss: `{best['_test_loss']:.4f}`",
        "",
        "## Runner-up",
        "",
    ]
    if runner_up:
        lines.extend(
            [
                f"- Run: `{runner_up['run_name']}`",
                f"- Model: `{runner_up['model']}`",
                f"- Best validation Q8: `{runner_up['_best_val']:.4f}`",
                f"- Test Q8 on CB513: `{runner_up['_test_q8']:.4f}`",
                f"- Test loss: `{runner_up['_test_loss']:.4f}`",
            ]
        )
    else:
        lines.append("- Not available yet")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Research is running sequentially and recording every experiment.",
            "- Phase 4 focuses on the strongest ResDil CNN1D candidates.",
            "- Full ledger is stored in `outputs/reports/run_ledger.csv`.",
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
        "--channels",
        str(exp["channels"]),
    ]
    started = iso_now()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = iso_now()
    report_path = output_dir / "report.json"
    if proc.returncode != 0 or not report_path.exists():
        append_ledger(
            {
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
                "width": exp["channels"],
                "split": "train_val_from_cullpdb__test_cb513",
                "save_dir": str(output_dir.relative_to(ROOT)),
                "report_path": "",
                "best_val_q8": "",
                "test_q8": "",
                "status": "failed",
                "notes": f"stderr_tail={proc.stderr[-300:]}",
            }
        )
        write_status()
        return {"ok": False}

    report = json.loads(report_path.read_text())
    append_ledger(
        {
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
            "width": exp["channels"],
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
    )
    write_status()
    return {"ok": True}


def build_phase4_runs() -> list[dict]:
    return [
        {
            "phase": "research_phase4",
            "run_name": "p4_01_resdil_b42_ce_none_c320_e18",
            "model": "resdil_cnn1d",
            "feature_set": "baseline42",
            "epochs": 18,
            "batch_size": 16,
            "lr": 0.0010,
            "weight_decay": 4e-6,
            "dropout": 0.22,
            "channels": 320,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 42,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_02_resdil_b42_ce_none_c320_e24",
            "model": "resdil_cnn1d",
            "feature_set": "baseline42",
            "epochs": 24,
            "batch_size": 16,
            "lr": 0.0009,
            "weight_decay": 2e-6,
            "dropout": 0.20,
            "channels": 320,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 42,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_03_resdil_b42_ce_none_c384_e24",
            "model": "resdil_cnn1d",
            "feature_set": "baseline42",
            "epochs": 24,
            "batch_size": 16,
            "lr": 0.00085,
            "weight_decay": 2e-6,
            "dropout": 0.20,
            "channels": 384,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 42,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_04_resdil_e46_ce_none_c320_e24",
            "model": "resdil_cnn1d",
            "feature_set": "extended46",
            "epochs": 24,
            "batch_size": 16,
            "lr": 0.0009,
            "weight_decay": 2e-6,
            "dropout": 0.20,
            "channels": 320,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 42,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_05_resdil_e46_ce_none_c384_e24",
            "model": "resdil_cnn1d",
            "feature_set": "extended46",
            "epochs": 24,
            "batch_size": 16,
            "lr": 0.00085,
            "weight_decay": 2e-6,
            "dropout": 0.20,
            "channels": 384,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 42,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_06_resdil_b42_focal_none_c320_e18",
            "model": "resdil_cnn1d",
            "feature_set": "baseline42",
            "epochs": 18,
            "batch_size": 16,
            "lr": 0.0010,
            "weight_decay": 4e-6,
            "dropout": 0.22,
            "channels": 320,
            "loss": "focal",
            "class_weighting": "none",
            "seed": 42,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_07_resdil_b42_ce_none_c320_e24_seed7",
            "model": "resdil_cnn1d",
            "feature_set": "baseline42",
            "epochs": 24,
            "batch_size": 16,
            "lr": 0.0009,
            "weight_decay": 2e-6,
            "dropout": 0.20,
            "channels": 320,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 7,
        },
        {
            "phase": "research_phase4",
            "run_name": "p4_08_resdil_b42_ce_none_c320_e24_seed13",
            "model": "resdil_cnn1d",
            "feature_set": "baseline42",
            "epochs": 24,
            "batch_size": 16,
            "lr": 0.0009,
            "weight_decay": 2e-6,
            "dropout": 0.20,
            "channels": 320,
            "loss": "ce",
            "class_weighting": "none",
            "seed": 13,
        },
    ]


def main() -> None:
    write_status()
    existing = existing_run_names()
    for exp in build_phase4_runs():
        if exp["run_name"] in existing:
            continue
        run_train(exp)
        existing.add(exp["run_name"])
    summary = {
        "updated_at": iso_now(),
        "phase": "phase4_targeted",
        "top10": rank_rows(load_ledger_rows())[:10],
    }
    (REPORTS_DIR / "phase4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
