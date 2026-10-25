"""Portable launcher for the supplied SACR-MVC experiment presets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = ("RGB-D", "Mfeat", "CIFAR10", "Hdigit", "ALOI-100", "CIFAR100")
VARIANTS = {
    "Full": {},
    "no_memory": {"enable_global_memory": False},
    "no_anchor": {"structural_anchor_epochs": 0},
    "no_pretrain_selection": {"pretrain_selection_mode": "none"},
    "no_guard": {"consensus_guard": False},
}

def preset(dataset, protocol="table"):
    if protocol != "table":
        raise ValueError("This package provides the table protocol only")
    spec = json.loads((ROOT / "configs" / (dataset + ".json")).read_text(encoding="utf-8"))
    return spec["table_run"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--protocol", choices=("table",), default="table")
    p.add_argument("--variant", choices=tuple(VARIANTS), default="Full")
    p.add_argument("--gpu", default="0", help="CUDA device index; use cpu for CPU execution")
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()
    if args.list_configs:
        for ds in DATASETS:
            spec = json.loads((ROOT / "configs" / (ds + ".json")).read_text(encoding="utf-8"))
            print(json.dumps(spec, ensure_ascii=False))
        return
    if args.dataset is None:
        p.error("--dataset is required unless --list-configs is used")
    try:
        cfg = preset(args.dataset, args.protocol)
    except ValueError as e:
        p.error(str(e))
    if args.protocol == "table" and args.dataset == "CIFAR10" and args.variant != "Full":
        p.error("This package provides module-ablation presets for ALOI-100 and Hdigit only")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SACR_")}
    env.update(cfg["env"])
    if args.data_root is None:
        if not args.dry_run:
            p.error("--data-root is required for training")
    else:
        env["SACR_DATASET_ROOT"] = str(args.data_root.expanduser().resolve())
    # Match the two-thread numerical-library setting used for the recorded runs.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "2"
    run_dir = args.output_dir.expanduser().resolve() / (args.dataset + "_" + args.protocol + "_" + str(cfg["args"]["seed"]) + "_" + args.variant)
    cli = dict(cfg["args"])
    cli.update(VARIANTS[args.variant])
    cli.update({"dataset": args.dataset, "gpu": args.gpu, "summary_path": str(run_dir / "summary.csv"), "save_model": args.save_model, "model_dir": str(run_dir / "models")})
    command = [sys.executable, "-u", str(ROOT / "src/train_final.py")]
    for name, value in cli.items():
        command += ["--" + name, str(value)]
    print(json.dumps({"command": command, "env": {k: v for k, v in env.items() if k.startswith("SACR_")}, "cwd": str(run_dir)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call(command, cwd=str(run_dir), env=env))

if __name__ == "__main__":
    main()
