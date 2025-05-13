import json
import pandas as pd
import matplotlib.pyplot as plt

# baseline
with open("metrics_baseline.json") as f:
    base = pd.DataFrame(json.load(f))
# возьмём первую (и единственную) строку
base_map50 = base.loc[0, "mAP50"]

# prune
df_prune = pd.read_json("metrics_prune.json")

# retrain
df_retrain = pd.read_json("metrics_retrain.json")

# График 1: baseline vs prune vs retrain для одного sparsity
s = 0.5  # пример sparsity, который нам интересен
pruned_m = df_prune[df_prune.sparsity == s].mAP50.values[0]
retrain_m = df_retrain[df_retrain.sparsity == s].mAP50.values[0]

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
plt.plot(df_prune.sparsity, df_prune.mAP50, marker="o", label="after prune")
plt.plot(df_retrain.sparsity, df_retrain.mAP50,
         marker="^", label="after retrain")
plt.title("mAP50 vs sparsity")
plt.xlabel("Sparsity")
plt.ylabel("mAP50")
plt.legend()
plt.tight_layout()
plt.show()
