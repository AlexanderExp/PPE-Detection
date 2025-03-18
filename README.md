# PPE-Detection

### Содержание репозитория
- Файл [EDA.ipynb](https://github.com/AlexanderExp/PPE-Detection/blob/main/EDA.ipynb) является ноутбуком с начальным анализом датасета и обоснованием выбора метрик для дальнейшей работы (КТ_1)
- Файл [Обзор_архитектур_и_сравнение_версий YOLO](https://github.com/AlexanderExp/PPE-Detection/blob/main/%D0%9E%D0%B1%D0%B7%D0%BE%D1%80_%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80_%D0%B8_%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D0%B9_YOLO.pdf) сосотоит из краткого обзора архитектур YOLO, DETR, Faster R-CNN, Center-Net, SOTA. После обзора можно найти сравнение моделей YOLO, сосредоточенного вокруг версий 8,9,10,11,12, и обоснование выбора YOLOv11m для этой работы (КТ_2)
- Файл [utils.py](https://github.com/AlexanderExp/PPE-Detection/blob/main/utils.py) содержит вспомогательные функции для обучения моделей, визуализации результатов (КТ_2)
- Файл [YOLOv11_Train.ipynb](https://github.com/AlexanderExp/PPE-Detection/blob/main/YOLOv11_Train.ipynb) состоит из ячеек с кодом обучения модели YOLOv11m для получения бейзлайна и метрик. Сохранена возможность обучения других версий YOLO для сравнения результатов (КТ_2)
- Файл [YOLOv11m_Results](https://github.com/AlexanderExp/PPE-Detection/blob/main/YOLOv11m_Results.pdf) содержит бейзлайн и результаты обучения модели YOLOv11m (КТ_2)
