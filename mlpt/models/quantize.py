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

def quantize_ptq(yolo_w, backend, calib_loader, dtype):
    torch.backends.quantized.engine = backend
    model = YOLO(yolo_w).model  # nn.Module
    model.eval()

    # Фьюзим Conv+BN+ReLU
    fuse_list = []
    for name, m in model.named_children():
        if isinstance(m, torch.nn.Sequential):
            for idx in range(len(m)-1):
                if isinstance(m[idx], torch.nn.Conv2d) and isinstance(m[idx+1], torch.nn.BatchNorm2d):
                    fuse_list.append(f"{name}.{idx},{name}.{idx+1}")
    if fuse_list:
        fuse_modules(model, fuse_list, inplace=True)

    # Назначаем конфиг
    model.qconfig = get_default_qconfig(backend)

    # Prepare (вставляет Observer-ы)
    prepared = prepare(model)
    # Калибровка
    with torch.inference_mode():
        for i, (imgs, _) in enumerate(calib_loader):
            prepared(imgs)
    # Convert FP32→INT8
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
    q_model = quantize_ptq(weights, backend, calib_loader, dtype)
    q_time = time.time() - t0

    # сохраняем INT8 веса
    os.makedirs("models", exist_ok=True)
    q_path = f"models/{p['model']['name']}_int8.pt"
    torch.save(q_model.state_dict(), q_path)
    print(f"[INFO] Saved quantized weights → {q_path}")

    # ========= Validation =========
    # чтобы не переписывать val-код, оборачиваем quant-модель в YOLO
    q_yolo = YOLO(q_model, task="detect")  # Ultralytics принимает nn.Module
    val_res = q_yolo.val(
        data=p["data"]["config"], imgsz=imgsz,
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
