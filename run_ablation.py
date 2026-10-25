"""Run the Table 4 module-ablation presets without changing the training code."""
import argparse
import json
import sys
import run


def main():
    spec = json.loads((run.ROOT / "configs/ablation_modules.json").read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=spec["datasets"])
    p.add_argument("--variant", choices=tuple(spec["variants"]))
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--gpu", default="0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--list-configs", action="store_true")
    args = p.parse_args()
    if args.list_configs:
        print(json.dumps(spec, indent=2))
        return
    if args.dataset is None or args.variant is None:
        p.error("--dataset and --variant are required unless --list-configs is used")
    run.VARIANTS.update(spec["variants"])
    command = [sys.argv[0], "--dataset", args.dataset, "--variant", args.variant, "--gpu", args.gpu]
    for name in ("data_root", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), value]
    for name in ("dry_run", "save_model"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-")]
    sys.argv = command
    run.main()


if __name__ == "__main__":
    main()
