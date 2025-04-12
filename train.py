from utils import train_and_validate_models
import pandas as pd
import json
import yaml
import argparse
import os
os.environ["MPLBACKEND"] = "agg"



def load_params(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    # Загружаем параметры эксперимента из файла конфигурации
    params = load_params(config_path)

    # Извлекаем необходимые параметры
    data_config = params["data"]["config"]
    models_to_train = {
        params["model"]["name"]: params["model"]["weights"],
    }
    epochs = params["training"]["epochs"]
    project_name = "runs/detect"  # Папка для сохранения результатов

    # Запуск обучения и валидации моделей
    results_list = train_and_validate_models(
        models_to_train, data_config, project_name, epochs)

    # Инициализация логгера для TensorBoard (логи будут сохранены в 'tensorboard_logs')
    from tb_logger import TensorBoardLogger
    tb_logger = TensorBoardLogger()

    # Логирование итоговых метрик для каждой модели
    for result in results_list:
        model_name = result["Model"]
        tb_logger.log_metrics(model_name, {
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "mAP50": result["mAP50"],
            "mAP50-95": result["mAP50-95"],
            "Training Time (s)": result["Training Time (s)"]
        }, step=epochs)  # Используем номер последней эпохи как шаг
    tb_logger.close()

    # Сохранение метрик в формате JSON (для DVC)
    with open("metrics.json", "w") as f:
        json.dump(results_list, f, indent=4)

    # Сохранение итоговых результатов также в CSV-файл
    results_df = pd.DataFrame(results_list)
    output_csv_path = "final_results.csv"
    results_df.to_csv(output_csv_path, index=False)

    print(
        f"Эксперимент завершён: результаты сохранены в metrics.json и {output_csv_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="params.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()
    main(args.config)
