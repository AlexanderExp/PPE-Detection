import time
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, base_log_dir="tensorboard_logs"):
        # Генерируем уникальную директорию для каждого запуска с использованием метки времени
        timestamp = int(time.time())
        self.log_dir = f"{base_log_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[INFO] Логи TensorBoard сохраняются в: {self.log_dir}")

    def log_metrics(self, model_name: str, metrics: dict, step: int = 0):
        """
        Логирует метрики эксперимента для конкретной модели.
        
        :param model_name: Имя модели, например, 'YOLOv11'.
        :param metrics: Словарь с метриками, например: {"Precision": 0.85, "Recall": 0.78, ...}.
        :param step: Шаг эксперимента (например, номер эпохи).
        """
        for metric, value in metrics.items():
            self.writer.add_scalar(f"{model_name}/{metric}", value, step)

    def close(self):
        self.writer.close()
