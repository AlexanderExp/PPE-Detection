from mlpt.utils.utils import train_and_validate_models
import time
import json
import yaml
import argparse
import os
import sys
# Добавляем корень репозитория в sys.path, предполагая, что данный файл находится в mlpt/models/
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")))

# Устанавливаем backend для matplotlib, если требуется
os.environ["MPLBACKEND"] = "agg"

# Далее обычные импорты:


def load_params(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    train_model(config_path)


def train_model(config_input):
    # Загружаем параметры из указанного YAML-файла
    if isinstance(config_input, dict):
        params = config_input
    else:
        # Иначе ожидаем путь к YAML
        params = load_params(config_input)

    data_config = params["data"]["config"]
    models_to_train = {params["model"]["name"]: params["model"]["weights"]}

    # Читаем секцию training и переопределяем параметры,
    # если они заданы в YAML-файле; иначе берём дефолтные значения.
    training_cfg = params.get("training", {})
    epochs = training_cfg.get("epochs", 1)
    batch = training_cfg.get("batch", 1)
    imgsz = training_cfg.get("imgsz", 320)
    mosaic = training_cfg.get("mosaic", 0.0)
    mixup = training_cfg.get("mixup", 0.0)
    augment = training_cfg.get("augment", False)
    fraction = training_cfg.get("fraction", 1.0)

    project_name = "runs/detect"  # Папка для сохранения результатов эксперимента

    print(
        f"[INFO] Параметры обучения: epochs={epochs}, batch={batch}, imgsz={imgsz}, mosaic={mosaic}, mixup={mixup}, augment={augment}, fraction={fraction}")

    # Запускаем обучение, передавая дополнительные параметры
    start_time = time.time()
    results_list = train_and_validate_models(
        models_to_train, data_config, project_name, epochs,
        batch=batch, imgsz=imgsz, mosaic=mosaic, mixup=mixup,
        augment=augment, fraction=fraction,
    )
    training_time = time.time() - start_time
    print(f"[INFO] Обучение завершилось за {training_time:.2f} сек.")

    # Логирование метрик через TensorBoard
    from mlpt.utils.tb_logger import TensorBoardLogger
    tb_logger = TensorBoardLogger()
    for result in results_list:
        model_name = result["Model"]
        tb_logger.log_metrics(model_name, {
            "Precision": result.get("Precision", 0),
            "Recall": result.get("Recall", 0),
            "mAP50": result.get("mAP50", 0),
            "mAP50-95": result.get("mAP50-95", 0),
            "Training Time (s)": result.get("Training Time (s)", training_time)
        }, step=epochs)
    tb_logger.close()

    # Сохраняем метрики в JSON-файл (для DVC)
    with open("metrics.json", "w") as f:
        json.dump(results_list, f, indent=4)

    print("Эксперимент завершён, результаты сохранены в metrics.json.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="params.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()
    main(args.config)
