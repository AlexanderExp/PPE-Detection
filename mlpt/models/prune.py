import argparse
import time
import json
import yaml
import os
import torch.nn.utils.prune as prune
import torch.nn as nn
from mlpt.modules.ultralytics.ultralytics import YOLO
from mlpt.utils.utils import wait_for_results_file


def load_params(cfg_path: str) -> dict:
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def prune_model_torch(model: nn.Module, amount: float):
    """
    Applies global unstructured L1 pruning to all Conv2d layers,
    then removes the reparameterization to make pruning permanent.
    """
    total, zero = 0, 0
    print(f"[INFO] Pruning model... (target sparsity={amount})", end="")
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
    # compute resulting sparsity
    for p in model.parameters():
        total += p.numel()
        zero += (p == 0).sum().item()
    sparsity = zero / total if total else 0
    print(f" achieved global sparsity={sparsity:.3f}")


def main(config_path: str):
    params = load_params(config_path)
    data_cfg = params["data"]["config"]
    weights = params["model"]["weights"]
    imgsz = params["training"]["imgsz"]
    device = params["training"].get("device", "cpu")
    sparsities = params["pruning"]["sparsity_levels"]

    results = []
    for s in sparsities:
        run_name = f"prune_{int(s*100)}"
        project = "runs/prune"
        print(f"\n=== Pruning sparsity {s:.2f} -> run: {run_name} ===")

        # load and prepare model
        model = YOLO(weights)
        # fuse conv+bn for correct pruning :contentReference[oaicite:0]{index=0}
        model.fuse()
        # apply pruning :contentReference[oaicite:1]{index=1}
        prune_model_torch(model.model, amount=s)

        # run validation
        start_time = time.time()
        val_results = model.val(
            data=data_cfg,
            imgsz=imgsz,
            project=project,
            name=run_name,
            save_json=False,  # JSON not needed here
            verbose=True
        )
        val_time = time.time() - start_time

        # locate results.csv if available
        run_folder = os.path.join(project, run_name)
        res_file = wait_for_results_file(
            run_folder, pattern="results.csv", timeout=10)
        if res_file:
            import pandas as pd
            df = pd.read_csv(res_file)
            last = df.iloc[-1]
            precision = last.get("metrics/precision(B)", 0)
            recall = last.get("metrics/recall(B)", 0)
            mAP50 = last.get("metrics/mAP50(B)", 0)
            mAP50_95 = last.get("metrics/mAP50-95(B)", 0)
        else:
            # fallback to returned object
            precision = getattr(val_results, "precision", 0)
            recall = getattr(val_results, "recall", 0)
            mAP50 = getattr(val_results, "mAP50", 0)
            mAP50_95 = getattr(val_results, "mAP50_95", 0)

        results.append({
            "sparsity": s,
            "Precision": precision,
            "Recall": recall,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "Validation Time (s)": round(val_time, 2)
        })
        print(
            f"[INFO] Results at sparsity {s}: mAP50={mAP50}, time={val_time:.1f}s")

    # save all prune metrics
    with open("metrics_prune.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[INFO] Pruning evaluation complete, saved metrics_prune.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to params YAML")
    args = parser.parse_args()
    main(args.config)
