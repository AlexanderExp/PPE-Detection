import argparse
import yaml
import optuna
from dvclive.optuna import DVCLiveCallback
from mlpt.models.train import train_model
import json


def objective(trial, config_path):
    # каждый раз читаем базовый конфиг
    with open(config_path) as f:
        params = yaml.safe_load(f)
    # overrides
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
    # запуск
    result = train_model(params)
    # возвращаем mAP50
    return result['mAP50']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    # сколько проб
    base = yaml.safe_load(open(args.config))
    n_trials = args.trials or base['tuning']['n_trials']

    study = optuna.create_study(
        direction="maximize", study_name="ppe_detection")
    study.optimize(
        lambda t: objective(t, args.config),
        n_trials=n_trials,
        callbacks=[DVCLiveCallback(metric_name="mAP50")]
    )

    # *** вставляем запись summary-файла ***
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
