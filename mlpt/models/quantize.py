import os
import time
import json
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.ao.quantization import (
    fuse_modules, get_default_qconfig, prepare, convert
)
from mlpt.modules.ultralytics.ultralytics import YOLO
from mlpt.utils.utils import wait_for_results_file
from itertools import islice
from tqdm import tqdm   
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.conv import Conv
from itertools import islice
from tqdm import tqdm
from torch.ao.quantization import fuse_modules, get_default_qconfig, prepare, convert

def load_params(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

# Простейший набор данных для калибровки


def build_calib_loader(val_txt, img_size, batch):
    with open(val_txt) as f:
        paths = [l.strip() for l in f]
    # Структура ImageFolder не нужна ‒ сделаем Dataset «из списка путей»

    class CalibDS(torch.utils.data.Dataset):
        def __len__(self): return len(paths)

        def __getitem__(self, i):
            from PIL import Image
            im = Image.open(paths[i]).convert("RGB")
            tfm = Compose([Resize((img_size, img_size)), ToTensor()])
            return tfm(im), torch.tensor(0)
    return DataLoader(CalibDS(), batch_size=batch, shuffle=False)

def quantize_ptq(weights, backend, calib_loader, calib_size, dtype):
    torch.backends.quantized.engine = backend
    model = YOLO(weights).model
    model.eval()

    # 1️⃣ Fuse только conv+bn (SiLU/act фьюзить нельзя)
    for m in model.modules():
        if isinstance(m, Conv):
            fuse_modules(m, ['conv', 'bn'], inplace=True)

    # 2️⃣ Назначаем qconfig и вставляем обсерверы
    model.qconfig = get_default_qconfig(backend)
    prepared = prepare(model)

    # 3️⃣ Калибровка первых calib_size изображений
    n_batches = calib_size // calib_loader.batch_size
    with torch.inference_mode():
        for imgs, _ in tqdm(islice(calib_loader, n_batches),
                            total=n_batches, desc="Calibrating"):
            prepared(imgs)

    # 4️⃣ Конвертация в INT8
    quantized = convert(prepared)
    return quantized

def main(cfg):
    p = load_params(cfg)
    backend = p["quantization"]["backend"]
    calib_sz = p["quantization"]["calib_size"]
    dtype = p["quantization"]["dtype"]
    val_txt = p["data"]["val"]
    imgsz = p["training"].get("imgsz", 320)
    weights = p["model"]["weights"]

    calib_loader = build_calib_loader(val_txt, imgsz, batch=16)

    print(f"[INFO] PTQ backend={backend}, dtype={dtype}, calib={calib_sz}")
    t0 = time.time()
    q_model = quantize_ptq(weights, backend, calib_loader, calib_sz, dtype)

    q_time = time.time() - t0

    # сохраняем INT8 веса
    os.makedirs("models", exist_ok=True)
    q_path = f"models/{p['model']['name']}_int8.pt"
    torch.save(q_model.state_dict(), q_path)
    print(f"[INFO] Saved quantized weights → {q_path}")

    # ========= Validation =========
    # загружаем обычную обёртку, а затем подменяем .model на квантизированный
    yolo = YOLO(weights, task="detect")   # usual wrapper
    yolo.model = q_model.to("cpu")        # INT8 model on CPU
    val_res = yolo.val(
        data=p["data"]["config"],
        imgsz=imgsz,
        device="cpu", batch=16, workers=0,
        project="runs/quant", name="int8", verbose=True
    )

    # Метрики (если Ultralytics ≥ 8.3 вернёт attr-ы)
    metrics = {
        "dtype": dtype,
        "Precision": getattr(val_res, "precision", 0),
        "Recall": getattr(val_res, "recall", 0),
        "mAP50": getattr(val_res, "mAP50", 0),
        "mAP50-95": getattr(val_res, "mAP50_95", 0),
        "Validation Time (s)": round(q_time, 2)
    }

    with open("metrics_quantize.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[INFO] Saved metrics_quantize.json:", metrics)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    main(a.parse_args().config)
