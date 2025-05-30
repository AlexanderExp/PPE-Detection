# В самом начале файла tune_optuna.py
from functools import lru_cache
import yaml
import argparse
import optuna
from dvclive.optuna import DVCLiveCallback
from mlpt.models.train import train_model
import json

# Обернём load_params в LRU-кеш
@lru_cache(maxsize=None)
def load_params(path: str) -> dict:
    """
    Кешированная загрузка YAML-конфига.
    При повторном вызове с тем же `path`
    будет возвращён уже распарсенный объект.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def objective(trial, config_path):
    # теперь load_params не будет читать файл каждый раз
    params = load_params(config_path)
    # переопределяем параметры обучения из секции tuning
    params['training']['batch'] = trial.suggest_categorical(
        "batch", params['tuning']['batch_opts'])
    params['training']['imgsz'] = trial.suggest_categorical(
        "imgsz", params['tuning']['imgsz_opts'])
    params['training']['lr'] = trial.suggest_float(
        "lr",
        float(params['tuning']['lr_min']),
        float(params['tuning']['lr_max']),
        log=True
    )
    # прогоняем тренировку
    result = train_model(params)
    return result['mAP50']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    # загружаем базовые параметры только один раз
    base = load_params(args.config)
    n_trials = args.trials or base['tuning']['n_trials']

    # создаём Study с любым storage (sqlite или pickle, как вам удобно)
    study = optuna.create_study(
        direction="maximize", study_name="ppe_detection"
    )
    study.optimize(
        lambda t: objective(t, args.config),
        n_trials=n_trials,
        callbacks=[DVCLiveCallback(metric_name="mAP50")]
    )

    # записываем сводку
    best = study.best_trial
    summary = {
        "best_batch": best.params["batch"],
        "best_imgsz": best.params["imgsz"],
        "best_lr": best.params["lr"],
        "best_mAP50": best.value
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Wrote summary metrics.json:", summary)


if __name__ == "__main__":
    main()
