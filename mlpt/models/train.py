from mlpt.utils.utils import train_and_validate_models
import time
import json
import yaml
import argparse
import os
import sys
import shutil

# Добавляем корень репозитория в sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")))

# Устанавливаем backend для matplotlib, если требуется
os.environ["MPLBACKEND"] = "agg"


def load_params(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    train_model(config_path)


def train_model(config_input):
    # Загружаем параметры
    if isinstance(config_input, dict):
        params = config_input
    else:
        params = load_params(config_input)

    data_config = params["data"]["config"]
    models_to_train = {params["model"]["name"]: params["model"]["weights"]}

    # Читаем параметры обучения
    training_cfg = params.get("training", {})
    epochs = training_cfg.get("epochs", 1)
    batch = training_cfg.get("batch", 1)
    imgsz = training_cfg.get("imgsz", 320)
    mosaic = training_cfg.get("mosaic", 0.0)
    mixup = training_cfg.get("mixup", 0.0)
    augment = training_cfg.get("augment", False)
    fraction = training_cfg.get("fraction", 1.0)
    workers = training_cfg.get("workers", 2)

    # Фиксированная папка для сохранения результатов
    project_name = "runs/detect"

    print(
        f"[INFO] Параметры обучения: epochs={epochs}, batch={batch}, imgsz={imgsz}, mosaic={mosaic}, mixup={mixup}, augment={augment}, fraction={fraction}")

    # Запуск обучения и валидации
    start_time = time.time()
    results_list = train_and_validate_models(
        models_to_train, data_config, project_name, epochs,
        batch=batch, imgsz=imgsz, mosaic=mosaic, mixup=mixup,
        augment=augment, fraction=fraction, workers=workers
    )
    training_time = time.time() - start_time
    print(f"[INFO] Обучение завершилось за {training_time:.2f} сек.")

    # Логирование метрик через TensorBoard
    from mlpt.utils.tb_logger import TensorBoardLogger
    tb_logger = TensorBoardLogger()
    for result in results_list:
        model_name = result["Model"]
        tb_logger.log_metrics(
            model_name,
            {
                "Precision": result.get("Precision", 0),
                "Recall": result.get("Recall", 0),
                "mAP50": result.get("mAP50", 0),
                "mAP50-95": result.get("mAP50-95", 0),
                "Training Time (s)": result.get("Training Time (s)", training_time)
            },
            step=epochs
        )
    tb_logger.close()

    # Сохранение метрик для DVC
    with open("metrics.json", "w") as f:
        json.dump(results_list, f, indent=4)
    print("Эксперимент завершён, результаты сохранены в metrics.json.")

    # Копируем лучшие веса в фиксированное место для DVC
    # Определяем имя модели и папку запуска
    model_name = list(models_to_train.keys())[0]
    run_dir = os.path.join(project_name, f"train_{model_name}")
    best_src = os.path.join(run_dir, "weights", "best.pt")
    dst_dir = "models"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{model_name}_best.pt")
    try:
        shutil.copy(best_src, dst)
        print(f"[INFO] Скопированы лучшие веса в {dst}")
    except FileNotFoundError:
        print(f"[WARNING] Файл {best_src} не найден, копирование пропущено.")

    return results_list[0] if results_list else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="params.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()
    main(args.config)
