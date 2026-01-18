# Jupyter Notebook'и проекта

## Обзор

Проект содержит 4 Jupyter Notebook'а для работы с данными The Guardian и обучения модели классификации текстов.

---

## 1. `WORKing (3)-Copy1.ipynb`
**Сбор данных через Guardian API**

### Назначение
Сбор статей из The Guardian по различным категориям через API.

### Основные шаги
- Конфигурация API-ключа и базового URL
- Сбор статей по 4 типам:
  - Analytical (`tone/analysis`)
  - Feature (`tone/features`)
  - Editorial (`tone/editorials`)
  - Review (`tone/reviews`)
- Пагинация: до 50 страниц по 200 статей на каждый тип
- Сохранение в `guardian_non_news.csv`

### Результат
~40,000 статей (10,000 каждого типа)

---

## 2. `cleaningTheGuard (1).ipynb`
**Очистка данных**

### Назначение
Предварительная обработка текстов статей для обучения модели.

### Основные шаги
- Загрузка `combined_guardian.csv`
- Функция `clean_fields()`:
  - Парсинг HTML с BeautifulSoup
  - Удаление URL-адресов
  - Удаление спецсимволов (оставляются только буквы, цифры, пробелы)
  - Нормализация пробелов и приведение к нижнему регистру
- Применение с прогресс-баром (tqdm) к 50,000 записям
- Сохранение в `cleaned_combined_guardian.csv`

---

## 3. `Obuchenie (2).ipynb`
**Обучение классификатора BERT**

### Назначение
Обучение модели BERT для классификации статей по 5 категориям.

### Категории
- News
- Analytical
- Feature
- Editorial
- Review

### Технологии
- **Модель**: BERT-base-uncased
- **Фреймворки**: PyTorch, Transformers (HuggingFace), Datasets
- **GPU**: CUDA (если доступна)

### Пайплайн
1. Загрузка и очистка данных (удаление NaN, пустых строк)
2. Кодирование меток (LabelEncoder)
3. Разделение: train/val/test (80/10/10) со стратификацией
4. Токенизация (max_length=256)
5. DataLoader (batch_size=16)
6. Обучение (3 эпохи, lr=2e-5, AdamW)
7. Валидация после каждой эпохи
8. Сохранение лучшей модели

### Результаты
- Validation Accuracy: **88.11%**
- Test Accuracy: **87.62%**

### Артефакты
- `best_model.pth` - веса лучшей модели
- `bert_category_classifier/` - директория с моделью и токенизатором
- `label_encoder.pkl` - энкодер меток

---

## 4. `Просмотр [oqAdDD].ipynb`
**Просмотр данных**

### Назначение
Быстрая проверка очищенных данных.

### Функционал
- Загрузка `cleaned_combined_guardian.csv`
- Вывод произвольных строк для проверки качества очистки

---

## Зависимости

### Основные библиотеки
```
pandas
torch
transformers
datasets
scikit-learn
beautifulsoup4
requests
tqdm
joblib
```

### Запуск
Убедитесь, что установлен CUDA для GPU-обучения (опционально).

---

## Порядок использования

1. **Сбор данных**: `WORKing (3)-Copy1.ipynb`
2. **Очистка**: `cleaningTheGuard (1).ipynb`
3. **Обучение**: `Obuchenie (2).ipynb`
4. **Проверка**: `Просмотр [oqAdDD].ipynb` (по необходимости)
