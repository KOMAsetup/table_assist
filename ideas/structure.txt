repo/
├── README.md
├── requirements.txt       # базовые зависимости
├── setup.py               # если делаем пакет
├── Dockerfile             # образ для всего проекта
├── docker-compose.yml     # если планируем несколько сервисов
├── data/                  # необработанные и подготовленные данные
├── notebooks/             # эксперименты и визуализация
├── pipelines/             # пайплайны обработки данных
│   └── main_pipeline.py
├── annotation_methods/    # разные способы аннотации
│   ├── method1/
│   │   ├── __init__.py
│   │   ├── annotate.py
│   │   └── utils.py
│   └── method2/
│       ├── __init__.py
│       ├── annotate.py
│       └── utils.py
├── api/                   # REST API
│   ├── __init__.py
│   ├── app.py             # точка входа FastAPI/Flask
│   └── routes.py          # эндпоинты
├── tests/                 # тесты для методов, пайплайнов и API
└── scripts/ 