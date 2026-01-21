# План реализации исследования

**Что vector representations reveal about publicistic writing: learning from mistakes**

---

## Обзор

**Общий срок**: 8–12 недель

**Цель**: Подготовить статью для конференции Dialogue по жанровой классификации публицистических текстов с фокусом на анализ ошибок

**Ключевая идея**: Ошибки классификации отражают реальную жанровую близость и градиентность жанровых границ

---

## Фаза 1: Базовая реализация (Недели 1–3)

### Задача 1.1: TF–IDF + Logistic Regression

**Срок**: 2–3 дня

**Шаги**:

1. Создать новый notebook: `models/tfidf_baseline.ipynb`

2. Реализовать пайплайн:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import GridSearchCV

   # TF-IDF векторизация
   vectorizer = TfidfVectorizer(
       max_features=10000,
       ngram_range=(1, 2),
       min_df=5,
       max_df=0.8
   )

   # Grid search для гиперпараметров
   param_grid = {'C': [0.1, 1, 10, 100]}
   ```

3. Обучить на train, валидировать на validation, протестировать на test

4. Сохранить:
   - Модель: `models/tfidf_lr.pkl`
   - Метрики: `results/tfidf_metrics.json`
   - Confusion matrix: `results/tfidf_confusion.npy`

5. Извлечь топ-50 слов для каждого жанра (коэффициенты модели)

**Результат**:
- Accuracy: ~75%
- Список лексических маркеров для каждого жанра

**Проверка**:
- [x] Модель обучена
- [x] Метрики сохранены
- [x] Confusion matrix визуализирована

---

### Задача 1.2: Лингвистические признаки

**Срок**: 4–5 дней

**Шаги**:

1. Создать notebook: `models/linguistic_features.ipynb`

2. Извлечение признаков (использовать spaCy):
   ```python
   import spacy
   from collections import Counter

   nlp = spacy.load("en_core_web_sm")

   def extract_linguistic_features(text):
       doc = nlp(text)

       features = {
           # Лексико-грамматические
           'type_token_ratio': len(set(doc)) / len(doc),
           'avg_sentence_length': np.mean([len(sent) for sent in doc.sents]),
           'modal_ratio': count_modals(doc) / len(doc),
           'first_person_ratio': count_pronouns(doc, person=1) / len(doc),
           'second_person_ratio': count_pronouns(doc, person=2) / len(doc),
           'third_person_ratio': count_pronouns(doc, person=3) / len(doc),

           # Дискурсивные
           'stance_markers_ratio': count_stance_markers(doc) / len(doc),
           'hedges_ratio': count_hedges(doc) / len(doc),
           'quotes_ratio': text.count('"') / len(text),
       }

       return features
   ```

3. Создать списки stance markers и hedges:
   ```python
   STANCE_MARKERS = [
       'arguably', 'reportedly', 'seemingly', 'apparently',
       'undoubtedly', 'clearly', 'obviously', 'evidently',
       'supposedly', 'presumably', 'ostensibly'
   ]

   HEDGES = [
       'perhaps', 'possibly', 'somewhat', 'rather',
       'quite', 'somewhat', 'relatively', 'comparatively'
   ]
   ```

4. Применить ко всем текстам (с progress bar)

5. Обучить Random Forest:
   ```python
   from sklearn.ensemble import RandomForestClassifier

   rf = RandomForestClassifier(
       n_estimators=200,
       max_depth=10,
       min_samples_split=5,
       random_state=42
   )
   ```

6. Feature importance: определить какие признаки наиболее важны

**Результат**:
- Accuracy: ~78%
- Список discriminative лингвистических признаков

**Проверка**:
- [x] Признаки извлечены без ошибок
- [x] Модель обучена
- [x] Feature importance проанализирован

**Проблемы**:
- spaCy может быть медленным на 50K текстов → использовать multiprocessing

---

### Задача 1.3: BERT fine-tuning

**Срок**: 3–4 дня

**Шаги**:

1. Создать notebook: `models/bert_finetuning.ipynb`

2. Использовать код из `alina/Obuchenie (2).ipynb` или:
   ```python
   from transformers import BertTokenizer, BertForSequenceClassification

   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   model = BertForSequenceClassification.from_pretrained(
       "bert-base-uncased",
       num_labels=5
   )
   ```

3. Гиперпараметры:
   - Learning rate: 2e-5
   - Epochs: 3
   - Batch size: 16
   - Max length: 256

4. Сохранить:
   - Модель: `models/bert_category_classifier/`
   - Метрики: `results/bert_metrics.json`
   - Confusion matrix: `results/bert_confusion_matrix.npy`

**Результат**:
- Accuracy: ~87-88%

**Проверка**:
- [x] BERT обучен
- [x] Метрики лучше TF-IDF (хотя бы на 0.5%)

**Проблемы**:
- BERT требует GPU → можно использовать Apple Silicon MPS или Google Colab
- Если не влезает: уменьшить batch size до 8

---

### Задача 1.4: Сводная таблица результатов

**Срок**: 1 день

**Шаги**:

1. Создать notebook: `results/baseline_comparison.ipynb`

2. Загрузить все метрики и сравнить:
   ```python
   results = {
       'TF-IDF + LR': load_metrics('results/tfidf_metrics.json'),
       'Linguistic + RF': load_metrics('results/linguistic_metrics.json'),
       'BERT': load_metrics('results/bert_metrics.json'),
   }

   comparison = pd.DataFrame(results).T
   ```

3. Визуализировать:
   - Bar chart с accuracy и F1
   - Confidence intervals (пока заглушки)

**Проверка**:
- [x] Все модели обучены
- [x] Таблица создана
- [x] BERT > TF–IDF > Linguistic

---

## Фаза 2: Анализ ошибок (Недели 4–6)

### Задача 2.1: Confusion matrices для всех моделей

**Срок**: 2 дня

**Шаги**:

1. Создать notebook: `analysis/confusion_matrices.ipynb`

2. Для каждой модели:
   ```python
   from sklearn.metrics import confusion_matrix
   import seaborn as sns

   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(10, 8))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

   plt.savefig('results/confusion_{model_name}.png')
   ```

3. Нормализовать confusion matrix (по строкам):
   ```python
   cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   ```

4. Сравнить паттерны ошибок между моделями

**Результат**:
- 3 confusion matrices (сырые + нормализованные)
- Выявлены жанровые пары с наибольшей путаницой

**Проверка**:
- [x] Все матрицы созданы
- [x] Identified top 3 error-prone genre pairs
по
---

### Задача 2.2: Выявление ключевых жанровых пар

**Срок**: 2 дня

**Шаги**:

1. Проанализировать confusion matrices

2. Для каждой модели определить топ-3 путающихся пар:
   ```python
   def get_top_confusions(cm, labels, top_n=3):
       confusions = []
       for i in range(len(labels)):
           for j in range(len(labels)):
               if i != j:
                   confusions.append({
                       'pair': (labels[i], labels[j]),
                       'count': cm[i, j]
                   })

       return sorted(confusions, key=lambda x: -x['count'])[:top_n]
   ```

3. Сравнить списки между моделями:
   - Какие пары стабильно путаются?
   - Есть ли различия между уровнями репрезентации?

**Ожидаемые результаты**:
- News ↔ Analysis (12%)
- Editorial ↔ Review (9%)
- Feature ↔ Analysis (8%)

**Проверка**:
- [x] Топ-3 пары идентифицированы
- [x] Есть пересечения между моделями

---

### Задача 2.3: Извлечение ошибочных примеров

**Срок**: 3 дня

**Шаги**:

1. Создать notebook: `analysis/error_examples.ipynb`

2. Для каждой ключевой жанровой пары:
   - Найти 10–15 примеров, где **все** модели ошибаются
   - Найти 5–10 примеров, где **только** BERT ошибается

3. Сохранить примеры:
   ```python
   error_examples = {
       'news_analysis': {
           'all_wrong': [indices],
           'bert_wrong_only': [indices]
       },
       'editorial_review': {...},
       'feature_analysis': {...}
   }
   ```

4. Вывести тексты с предсказаниями:
   ```python
   def print_error_example(idx, text, true_label, predictions):
       print(f"True: {true_label}")
       for model, pred in predictions.items():
           print(f"{model}: {pred} {'✓' if pred == true_label else '✗'}")
       print(f"Text: {text[:500]}...")
       print("-" * 80)
   ```

**Проверка**:
- [x] Собрано 30–50 ошибочных примеров
- [x] Примеры сохранены в CSV/JSON

---

### Задача 2.4: Качественный лингвистический анализ

**Срок**: 5–7 дней

**Шаги**:

1. Создать document: `analysis/qualitative_analysis.md`

2. Для каждого ошибочного примера проанализировать:

   **Модальность**:
   - Какие модальные глаголы используются?
   - Как выражена авторская позиция?
   - Есть ли hedges?

   **Дискурсивные маркеры**:
   - Stance markers (arguably, reportedly, etc.)
   - Цитаты и reporting verbs
   - Метатекстовые комментарии

   **Структура**:
   - Нарративная или логическая?
   - Есть ли ярко выраженная аргументация?
   - Наличие примеров, иллюстраций

   **Оценочность**:
   - Оценочная лексика (excellent, terrible, etc.)
   - Эмоционально окрашенные слова

3. Идентифицировать паттерны:
   - Что общего у текстов, путающих News и Analysis?
   - Почему Editorial путается с Review?

4. Связать с жанровой теорией:
   - Bhatia (1993) — genre as rhetorical action
   - Swales (2004) — genre as communicative event
   - Сравнить с теоретическими ожиданиями

**Результат**:
- Документ 5–10 страниц с качественным анализом
- 3–5 ключевых паттернов ошибок

**Проверка**:
- [x] Проанализировано минимум 15 примеров
- [x] Выявлены лингвистические паттерны
- [x] Связь с теорией установлена

---

## Фаза 3: Интерпретация BERT (Недели 7–8)

### Задача 3.1: Attention extraction

**Срок**: 3–4 дня

**Шаги**:

1. Создать notebook: `analysis/attention_analysis.ipynb`

2. Извлечь attention weights для [CLS] токена:
   ```python
   def extract_attention(model, tokenizer, text):
       inputs = tokenizer(text, return_tensors="pt")
       outputs = model(**inputs, output_attentions=True)

       # Average attention across all heads and layers
       attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
       cls_attention = attentions[:, :, :, 0, :]  # Attention to [CLS]
       avg_attention = cls_attention.mean(dim=(0, 1, 2))  # Average over layers, batch, heads

       return avg_attention
   ```

3. Визуализировать top-attended words:
   ```python
   def visualize_attention(text, attention, tokenizer, top_n=10):
       tokens = tokenizer.tokenize(text)
       top_indices = attention.argsort()[-top_n:][::-1]

       for idx in top_indices:
           print(f"{tokens[idx]}: {attention[idx]:.3f}")
   ```

**Результат**:
- Attention weights для 20–30 примеров (правильных + ошибочных)

**Проверка**:
- [x] Attention извлечён
- [x] Визуализация работает

---

### Задача 3.2: Сравнение attention patterns

**Срок**: 2–3 дня

**Шаги**:

1. Сравнить attention для разных жанров:
   - Какие слова типичны для News?
   - Какие для Editorial?
   - Какие для Review?

2. Для ошибочных примеров:
   - На что модель смотрит, когда ошибается?
   - Есть ли систематические паттерны?

3. Пример гипотезы:
   - Если путает News и Analysis → смотрит на фактуальные слова, но пропускает дискурсивные маркеры

**Результат**:
- Таблица: топ-10 слов для каждого жанра по attention
- Анализ ошибочных примеров через призму attention

**Проверка**:
- [x] Выявлены жанровые паттерны attention
- [x] Ошибки интерпретированы

---

### Задача 3.3: SHAP/LIME (опционально)

**Срок**: 3–4 дня (если есть время)

**Шаги**:

1. Создать notebook: `analysis/shap_analysis.ipynb`

2. Использовать SHAP для объяснения предсказаний:
   ```python
   import shap

   explainer = shap.Explainer(model, tokenizer)
   shap_values = explainer(texts)

   shap.plots.text(shap_values[0])
   ```

3. Сравнить SHAP values с attention:
   - Есть ли корреляция?
   - Что лучше объясняет ошибки?

**Результат**:
- SHAP visualizations для 10–15 примеров

**Проверка**:
- [ ] SHAP работает
- [ ] Результаты интерпретируемы

**Примечание**: Можно пропустить, если не хватает времени

---

## Фаза 4: Статистическая валидация и написание (Недели 9–12)

### Задача 4.1: McNemar's test

**Срок**: 2 дня

**Шаги**:

1. Создать notebook: `analysis/statistical_tests.ipynb`

2. Сравнить модели попарно:
   ```python
   from statsmodels.stats.contingency_tables import mcnemar

   def compare_models(y_true, y_pred1, y_pred2):
       # Create contingency table
       #        Model2 Correct | Model2 Wrong
       # Model1 Correct |     a      |      b
       # Model1 Wrong   |     c      |      d

       a = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
       b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
       c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
       d = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))

       result = mcnemar([[a, b], [c, d]], exact=True)

       return result
   ```

3. Сравнить:
   - BERT vs TF–IDF
   - BERT vs Linguistic
   - TF–IDF vs Linguistic

**Результат**:
- p-values для каждой пары
- Какие различия статистически значимы?

**Проверка**:
- [ ] Все пары сравнены
- [ ] Результаты интерпретируемы

---

### Задача 4.2: Bootstrap confidence intervals

**Срок**: 2 дня

**Шаги**:

1. Bootstrap для accuracy и F1:
   ```python
   from sklearn.utils import resample

   def bootstrap_metric(y_true, y_pred, metric, n_iterations=1000):
       scores = []
       for _ in range(n_iterations):
           y_true_bs, y_pred_bs = resample(y_true, y_pred)
           score = metric(y_true_bs, y_pred_bs)
           scores.append(score)

       return np.mean(scores), np.percentile(scores, [2.5, 97.5])
   ```

2. Для каждой модели:
   ```python
   mean, ci = bootstrap_metric(y_test, y_pred, accuracy_score)
   print(f"Accuracy: {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
   ```

**Результат**:
- Таблица с confidence intervals для всех метрик

**Проверка**:
- [ ] 95% CI посчитаны для всех моделей
- [ ] Результаты добавлены в таблицу

---

### Задача 4.3: Inter-annotator agreement

**Срок**: 3–4 дня

**Шаги**:

1. Подготовить выборку:
   - Случайные 300 текстов
   - Экспортировать в CSV/Google Sheets

2. Найти второго аннотатора (или сделать самому с перерывом)

3. Разметить жанры независимо

4. Рассчитать Cohen's Kappa:
   ```python
   from sklearn.metrics import cohen_kappa_score

   kappa = cohen_kappa_score(annotator1, annotator2)
   print(f"Cohen's Kappa: {kappa:.3f}")
   ```

5. Если κ < 0.7:
   - Проанализировать разногласия
   - Уточнить критерии разметки

**Результат**:
- Cohen's Kappa > 0.7
- Валидация ground truth

**Проверка**:
- [ ] Два независимых аннотатора
- [ ] Kappa рассчитан
- [ ] Если низкий — разногласия проанализированы

---

### Задача 4.4: Написание статьи

**Срок**: 2–3 недели

**Шаги**:

1. Создать document: `paper/dialogue_2026_paper.tex` или `.md`

2. Структура (согласно шаблону Dialogue):

   **Abstract** (150–200 слов):
   - Проблема
   - Метод
   - Основные результаты
   - Выводы

   **1. Introduction** (2–3 страницы):
   - Актуальность жанровой классификации
   - Проблема: недостаток интерпретируемости
   - RQs
   - Вклад

   **2. Related Work** (2 страницы):
   - Жанровая теория (Bhatia, Swales)
   - NLP подходы к жанровой классификации
   - Анализ ошибок в NLP

   **3. Data** (1 страница):
   - The Guardian corpus
   - Жанровые метки
   - Inter-annotator agreement

   **4. Methods** (3–4 страницы):
   - 4.1 TF–IDF + LR
   - 4.2 Linguistic features + RF
   - 4.3 BERT fine-tuning
   - 4.4 Statistical validation

   **5. Results** (2–3 страницы):
   - 5.1 Overall performance (Table 1)
   - 5.2 Confusion matrices
   - 5.3 Error patterns

   **6. Discussion** (3–4 страницы):
   - 6.1 Genre boundaries as gradient
   - 6.2 Error interpretation
   - 6.3 Attention analysis
   - 6.4 Limitations

   **7. Conclusion** (1 страница):
   - Выводы
   - Future work

   **References**

3. Требования Dialogue:
   - 8–12 страниц
   - LaTeX шаблон с конференции
   - Deadline: обычно апрель-май

**Результат**:
- Черновик статьи
- Все графики и таблицы

**Проверка**:
- [ ] Структура соответствует шаблону
- [ ] Все RQs отвечены
- [ ] Лимит страниц соблюдён
- [ ] References оформлены правильно

---

## Ресурсы и инструменты

### Необходимое ПО

```bash
# Python
python 3.9+

# NLP
pip install spacy transformers torch datasets scikit-learn

# Визуализация
pip install matplotlib seaborn shap

# Статистика
pip install statsmodels scipy

# Jupyter
pip install jupyter ipywidgets

# spaCy модель
python -m spacy download en_core_web_sm
```

### Железо

- **GPU**: NVIDIA GPU с 8+ GB VRAM или Apple Silicon MPS (для BERT)
- **RAM**: 16+ GB
- **Disk**: 10+ GB свободных

Если нет GPU:
- Использовать Google Colab (бесплатно)
- Kaggle Notebooks (бесплатно)

### Данные

- `cleaned_combined_guardian.csv` (уже есть)
- 50K текстов, 5 жанров

---

## Milestones (промежуточные контрольные точки)

| Неделя | Milestone | Что должно быть готово |
|--------|-----------|------------------------|
| 1–3 | Baseline models | TF–IDF, Linguistic, BERT обучены |
| 4–6 | Error analysis | Confusion matrices, качественный анализ |
| 7–8 | Interpretation | Attention visualization |
| 9–10 | Validation | McNemar, bootstrap, inter-annotator |
| 11–12 | Paper draft | Черновик статьи готов |

---

## Риски и митигация

### Риск 1: Недостаточно времени

**Митигация**:
- Приоритезация: BERT + анализ ошибок → минимум для acceptance
- Опционально: SHAP (если не успеваем)

### Риск 2: Низкая inter-annotator agreement

**Митигация**:
- Уточнить критерии разметки до начала
- Если κ < 0.7 → честно написать в limitations

### Риск 3: BERT не влезает в GPU

**Митигация**:
- Использовать Colab
- Gradient accumulation
- DistilBERT (меньше, чуть хуже)

### Риск 4: Нет явных паттернов в ошибках

**Митигация**:
- Расширить выборку ошибочных примеров
- Фокус на качественном анализе
- Честно написать: "mixed results"

---

## Checklist перед submission

### Модели:
- [x] TF–IDF обучен и сохранён
- [x] Linguistic features извлечены
- [x] BERT обучен

### Анализ:
- [x] Confusion matrices для всех моделей
- [x] 30–50 ошибочных примеров собрано
- [x] Качественный анализ проведён
- [x] Attention visualization

### Статистика:
- [ ] McNemar's test
- [ ] Bootstrap CIs
- [ ] Inter-annotator agreement

### Статья:
- [ ] Abstract написан
- [ ] Introduction с RQs
- [ ] Methods детально описаны
- [ ] Results с таблицами/графиками
- [ ] Discussion с интерпретацией
- [ ] Conclusion
- [ ] References
- [ ] Лимит страниц соблюдён
- [ ] LaTeX/LaTeX шаблон соблюдён

---

## Полезные ссылки

### Конференция Dialogue:
- https://dialogue-conf.org/
- Шаблоны: https://dialogue-conf.org/submission.html

### Жанровая теория:
- Bhatia, V. K. (1993). Analysing Genre: Language Use in Professional Settings
- Swales, J. M. (2004). Research Genres: Explorations and Applications

### NLP инструменты:
- HuggingFace: https://huggingface.co/docs
- spaCy: https://spacy.io/usage
- SHAP: https://shap.readthedocs.io/

### Статистика:
- McNemar test: https://en.wikipedia.org/wiki/McNemar%27s_test
- Bootstrap: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

---

## Дополнительные идеи (если останется время)

1. **Cross-lingual**: то же самое на русском тексте (Лента.ру, Газета.ру)
2. **Diachronic**: сравнить 2015–2017 vs 2023–2025
3. **Probing classifiers**: какие слои BERT кодируют жанровую информацию?
4. **Human evaluation**: показать людям ошибочные примеры, спросить — какой жанр?

---

**Удачи с исследованием! 🚀**
