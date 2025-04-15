# PPE-Detection

### Содержание репозитория
- Файл [EDA.ipynb](https://github.com/AlexanderExp/PPE-Detection/blob/main/notebooks/EDA.ipynb) является ноутбуком с начальным анализом датасета и обоснованием выбора метрик для дальнейшей работы (КТ_1)
- Файл [Обзор_архитектур_и_сравнение_версий YOLO](https://github.com/AlexanderExp/PPE-Detection/blob/main/%D0%9E%D0%B1%D0%B7%D0%BE%D1%80_%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80_%D0%B8_%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D0%B9_YOLO.pdf) сосотоит из краткого обзора архитектур YOLO, DETR, Faster R-CNN, Center-Net, SOTA. После обзора можно найти сравнение моделей YOLO, сосредоточенного вокруг версий 8,9,10,11,12, и обоснование выбора YOLOv11m для этой работы (КТ_2)
- Файл [YOLOv11_Train.ipynb](https://github.com/AlexanderExp/PPE-Detection/blob/main/notebooks/YOLOv11_Train.ipynb) состоит из ячеек с кодом обучения модели YOLOv11m для получения бейзлайна и метрик. Сохранена возможность обучения других версий YOLO для сравнения результатов (КТ_2)
- Файл [YOLOv11m_Results](https://github.com/AlexanderExp/PPE-Detection/blob/main/YOLOv11m_Results.pdf) содержит бейзлайн и результаты обучения модели YOLOv11m (КТ_2)


### Установка необходимого окружения

python -m venv venv

pip install -r requirements.txt

pip install -e .

dvc init

dvc exp run

tensorboard --logdir runs\detect\train_YOLOv11{NumberOfRun}



### Структура репозитория
```plaintext
PPE-Detection/
├── mlpt/
│   ├── datamodules/        # (Опционально) Логика загрузки/обработки данных
│   │   └── __init__.py
│   ├── models/             # Скрипты, связанные с обучением
│   │   ├── train.py        # Основной скрипт тренировки модели
│   │   └── __init__.py
│   ├── modules/            # Дополнительные модули и форкнутые библиотеки
│   │   └── ultralytics/    # Локальный форк ultralytics (подключается через Git submodule или editable install)
│   │       ├── __init__.py  # Экспортирует основные классы
│   │       └── ultralytics/
│   │           ├── __init__.py
│   │           ├── engine/
│   │           │   ├── __init__.py
│   │           │   └── model.py   # Определение класса Model, YOLO и т.д.
│   │           └── models/
│   │               ├── __init__.py
│   │               └── yolo/
│   │                   ├── __init__.py   # Относительные импорты внутри форка
│   ├── utils/              # Утилиты и логгеры
│   │   ├── utils.py        # Функции update, wait_for_results_file, train_and_validate_models, aggregate_results, plot_results
│   │   ├── tb_logger.py    # Логгер для TensorBoard
│   │   └── __init__.py     # Экспорт функций для удобного импорта
├── notebooks/              # Ноутбуки для анализа и отладки (EDA, обучение)
│   ├── EDA.ipynb
│   └── YOLOv11_Train.ipynb
├── sh17-dataset/           # Каталог с данными (скачан, распакован, обновлённые списки)
├── runs/                   # Результаты экспериментов
├── tensorboard_logs/       # Логи TensorBoard
├── dvc.yaml                # DVC-пайплайн для экспериментов (тренировка)
├── .dvcignore              # Файл, где указаны исключения для DVC (например, игнорировать mlpt/modules/ultralytics/)
├── .gitignore              # Игнорирование файлов Git
├── params.yaml             # Основная конфигурация эксперимента (baseline)
├── params_light_weight.yaml# Конфигурация для быстрого тестового прогона 
├── requirements.txt        # Все зависимости с фиксированными версиями
├── setup.py                # Скрипт установки пакета в editable‑режиме
└── README.md               # Документация проекта

