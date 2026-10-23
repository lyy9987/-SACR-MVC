"""Evaluate a final-model checkpoint saved by run.py --save-model."""
import argparse
import json
import os
from pathlib import Path
import sys
from run import ROOT, DATASETS, preset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--protocol", choices=("table", "frozen"), default="table")
    p.add_argument("--seed", type=int)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    try:
        cfg = preset(args.dataset, args.protocol, args.seed)
    except ValueError as e:
        p.error(str(e))
    for name in tuple(os.environ):
        if name.startswith("SACR_"):
            del os.environ[name]
    os.environ.update(cfg["env"])
    os.environ["SACR_DATASET_ROOT"] = str(args.data_root.resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import numpy as np
    import train_final as T
    from network import Network
    from dataloader import MultiviewData
    from dataset_metadata import known_cluster_count
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Checkpoints contain Python configuration objects; load only your own files.
    payload = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    if payload.get("method") != "SACR-MVC" or payload.get("dataset") != args.dataset:
        p.error("Checkpoint method/dataset mismatch")
    config = payload["config"]
    if int(config["seed"]) != int(cfg["args"]["seed"]):
        p.error("Checkpoint seed differs from the selected protocol")
    T.set_seed(int(config["seed"]))
    data = MultiviewData(args.dataset, device)
    num_views, num_samples = len(data.data_views), len(data)
    num_clusters = known_cluster_count(args.dataset)
    input_sizes = np.asarray([v.shape[1] for v in data.data_views], dtype=int)
    model = Network(
            num_views,
            num_samples,
            num_clusters,
            input_sizes,
            config["feature_dim"],
            memory_positive_k=config["global_memory_positive_k"],
            memory_negative_k=config["global_memory_negative_k"],
            # This is deliberately activated only after reconstruction
            # pretraining and its label-free checkpoint selection have finished.
            normalize_fusion_features=False,
            enable_semantic_projector=(
                config["enable_semantic_projector"]
                or config["adaptive_semantic_projector"]
            ),
            semantic_projector_hidden_dim=config["semantic_projector_hidden_dim"],
            semantic_projector_scale=(
                0.0
                if config["adaptive_semantic_projector"]
                else config["semantic_projector_scale"]
            ),
            shared_private_scale=(
                0.0
                if config["adaptive_shared_private"]
                else config["shared_private_scale"]
            ),
            enable_shared_private=config["adaptive_shared_private"],
            shared_private_hidden_dim=config["shared_private_hidden_dim"],
            context_aware_moe=config["context_aware_moe"],
            moe_shared_experts=config["moe_shared_experts"],
            moe_private_experts=config["moe_private_experts"],
            moe_top_k=config["moe_top_k"],
            enable_canonical_correlation_head=(
                float(config["canonical_correlation_weight"]) > 0.0
            ),
            canonical_correlation_dim=config["canonical_correlation_dim"],
            enable_multilevel_heads=config["enable_multilevel_heads"],
            multilevel_projection_dim=config["multilevel_projection_dim"],
        ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    route, metrics = T.fixed_route_clustering(model, data, config, args.dataset)
    print(json.dumps({"dataset": args.dataset, "metrics": metrics, "route": route}, ensure_ascii=False))

if __name__ == "__main__":
    main()
