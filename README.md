# МФТИ и Альфа - Отклик на кредитный оффер

**Задача.** Бинарная классификация: примет ли клиент предложение банка по овердрафту.  
Основная метрика — **ROC-AUC**

---

## Структура проекта

```
data/                        # Исходные данные (train / test)
├── train_apps.csv
├── test_apps.csv
└── sample_submission.csv

src/model/                   # Исходный код
├── eda.py                   # Разведочный анализ данных (распределения, корреляции, выбросы)
├── preprocessing.py         # Очистка, заполнение пропусков и прочая предобработка
├── features.py              # Выведение признаков (rate_spread, limit_ratio и др.)
├── feature_selection.py     # Отбор признаков (Gain, Permutation, SHAP, VIF)
├── baseline.py              # Базовая модель логистической регрессии
├── models.py                # LightGBM, XGBoost, CatBoost, RandomForest + сравнение
├── tuning.py                # Оптимизация гиперпараметров через Optuna
└── final_model.py           # Финальная модель + калибровка + SHAP + предсказания

notebooks/
└── pipeline.ipynb           # Основной пайплайн от EDA до загрузки решения

models/                      # Сохранённые модели (.pkl)
outputs/                     # Предсказания и файлы для загрузки в систему
```

---

## Как использовать

Запуск пайплайна — через ноутбук `notebooks/pipeline.ipynb` (шаг за шагом: EDA → препроцессинг → признаки → модели → тюнинг → финальное обучение → предсказания).

Основные модули можно вызывать и по отдельности:

```python
from src.model.preprocessing import Preprocessor
from src.model.features import create_features
from src.model.models import train_models
from src.model.tuning import tune_hyperparams
from src.model.final_model import train_final_model, generate_predictions
```

---

## Пайплайн (коротко)

1. **EDA** — смотрим на данные, целевую переменную, корреляции, выбросы.
2. **Preprocessing** — дропаем лишние колонки, заполняем пропуски, кодируем категории, меняем размерности.
3. **Feature Engineering** — добавляем производные признаки (спред ставки, отношения лимитов, активность клиента и т.д.).
4. **Feature Selection** — оставляем только важное (Gain, Permutation Importance, SHAP, проверка мультиколлинеарности).
5. **Baseline** — Логистическая регрессия для нижней планки.
6. **Model Training** — обучаем и сравниваем LightGBM, XGBoost, CatBoost, RandomForest (StratifiedKFold, 5 фолдов).
7. **Tuning** — Optuna подбирает гиперпараметры лучшей модели.
8. **Final Model** — обучаем на всех данных, калибруем вероятности, интерпретируем через SHAP.
9. **Prediction** — генерируем файл `outputs/predictions.csv` для загрузки в систему.

---

## Зависимости

`pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `matplotlib`, `seaborn`, `scipy`, `shap` (опционально).
