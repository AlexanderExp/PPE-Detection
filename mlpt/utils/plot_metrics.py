import json
import pandas as pd
import matplotlib.pyplot as plt

# baseline
with open("metrics_baseline.json") as f:
    base = pd.DataFrame(json.load(f))
# возьмём первую (и единственную) строку
base_map50 = base.loc[0, "mAP50"]

# prune
prune = pd.read_json("metrics_prune.json")

# retrain
retrain = pd.read_json("metrics_retrain.json")

# quantize
quant = pd.read_json("metrics_quantize.json")

# График 1: baseline vs prune vs retrain для одного sparsity
s = 0.5  # пример sparsity, который нам интересен
pruned_m = prune[prune.sparsity == s].mAP50.values[0]
retrain_m = retrain[retrain.sparsity == s].mAP50.values[0]

stages = ["baseline", f"prune_{int(s*100)}%", f"retrain_{int(s*100)}%"]
maps = [base_map50, pruned_m, retrain_m]

plt.figure()
plt.plot(stages, maps, marker="o")
plt.title(f"mAP50 @ sparsity={s}")
plt.ylabel("mAP50")
plt.xlabel("Stage")
plt.tight_layout()
plt.show()

# График 2: полный профиль prune и retrain
plt.figure()
plt.plot(prune.sparsity, prune.mAP50, marker="o", label="after prune")
plt.plot(retrain.sparsity, retrain.mAP50,
         marker="^", label="after retrain")
plt.title("mAP50 vs sparsity")
plt.xlabel("Sparsity")
plt.ylabel("mAP50")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(["baseline"], [base_map50])
plt.bar(quant.dtype, quant.mAP50)          # INT8
plt.bar([f"prune_{int(s*100)}%" for s in prune.sparsity], prune.mAP50)
plt.ylabel("mAP50")
plt.title("Baseline vs Prune vs Quant")
plt.tight_layout()
plt.show()
