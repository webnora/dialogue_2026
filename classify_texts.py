import pandas as pd
import joblib
import spacy
import numpy as np
import sys
import warnings

# Подавляем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# --- Начало скопированного кода из 02_linguistic_features.py ---

# Списки для stance-маркеров и хеджей
STANCE_MARKERS = [
    'arguably', 'reportedly', 'seemingly', 'apparently',
    'undoubtedly', 'clearly', 'obviously', 'evidently',
    'supposedly', 'presumably', 'ostensibly', 'arguably'
]

HEDGES = [
    'perhaps', 'possibly', 'somewhat', 'rather',
    'quite', 'relatively', 'comparatively', 'somewhat'
]

MODAL_VERBS = [
    'can', 'could', 'may', 'might', 'must',
    'should', 'ought', 'would', 'shall'
]

REPORTING_VERBS = [
    'said', 'says', 'say', 'told', 'tells', 'tell',
    'claimed', 'claims', 'claim', 'stated', 'states', 'state',
    'reported', 'reports', 'report', 'announced', 'announces'
]

def extract_linguistic_features(text, nlp):
    """
    Извлекает лингвистически мотивированные признаки из текста.
    """
    doc = nlp(text)
    features = {}
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    if len(tokens) > 0:
        features['type_token_ratio'] = len(set(tokens)) / len(tokens)
    else:
        features['type_token_ratio'] = 0
    sentences = list(doc.sents)
    if len(sentences) > 0:
        sent_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = np.mean(sent_lengths)
    else:
        features['avg_sentence_length'] = 0
    first_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['i', 'we', 'me', 'us', 'my', 'our'])
    second_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['you', 'your'])
    third_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their'])
    total_tokens = len(doc)
    if total_tokens > 0:
        features['first_person_ratio'] = first_person / total_tokens
        features['second_person_ratio'] = second_person / total_tokens
        features['third_person_ratio'] = third_person / total_tokens
    else:
        features['first_person_ratio'] = 0
        features['second_person_ratio'] = 0
        features['third_person_ratio'] = 0
    modal_count = sum(1 for token in doc if token.text.lower() in MODAL_VERBS)
    features['modal_ratio'] = modal_count / total_tokens if total_tokens > 0 else 0
    stance_count = sum(1 for token in doc if token.text.lower() in STANCE_MARKERS)
    features['stance_markers_ratio'] = stance_count / total_tokens if total_tokens > 0 else 0
    hedge_count = sum(1 for token in doc if token.text.lower() in HEDGES)
    features['hedges_ratio'] = hedge_count / total_tokens if total_tokens > 0 else 0
    quote_count = text.count('"')
    features['quotes_ratio'] = quote_count / total_tokens if total_tokens > 0 else 0
    reporting_count = sum(1 for token in doc if token.text.lower() in REPORTING_VERBS)
    features['reporting_verbs_ratio'] = reporting_count / total_tokens if total_tokens > 0 else 0
    return features

# --- Конец скопированного кода ---


def classify_texts():
    """
    Классифицирует 100 текстов с помощью двух моделей: TF-IDF и лингвистической.
    Сохраняет результаты в CSV файл.
    """
    print("Запуск классификации 100 текстов (исправленная версия)...")

    # --- 1. Загрузка данных ---
    try:
        texts_df = pd.read_csv("cnn_dailymail_100_texts.csv", header=None, names=['text'])
        texts = texts_df['text'].tolist()
        print(f"✓ [1/5] Загружено {len(texts)} текстов.")
    except FileNotFoundError:
        print("Ошибка: Файл 'cnn_dailymail_100_texts.csv' не найден.")
        return

    # --- 2. Загрузка моделей ---
    try:
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        tfidf_model = joblib.load("models/tfidf_lr.pkl")
        linguistic_model = joblib.load("models/linguistic_rf.pkl")
        nlp = spacy.load('en_core_web_sm')
        print("✓ [2/5] Все модели и spacy успешно загружены.")
    except FileNotFoundError as e:
        print(f"Ошибка при загрузке модели: {e}.")
        return
    except OSError:
        print("Ошибка: Не найдена модель spaCy 'en_core_web_sm'.")
        print("Пожалуйста, установите ее командой: python -m spacy download en_core_web_sm")
        return

    # --- 3. Классификация с помощью TF-IDF ---
    print("→ [3/5] Классификация с помощью TF-IDF...")
    texts_tfidf = tfidf_vectorizer.transform(texts)
    tfidf_predictions = tfidf_model.predict(texts_tfidf)
    print("✓ TF-IDF классификация завершена.")

    # --- 4. Классификация с помощью лингвистических признаков ---
    print("→ [4/5] Классификация с помощью лингвистических признаков (это может занять время)...")
    linguistic_features_list = [extract_linguistic_features(text, nlp) for text in texts]
    features_df = pd.DataFrame(linguistic_features_list)
    linguistic_predictions = linguistic_model.predict(features_df)
    print("✓ Лингвистическая классификация завершена.")

    # --- 5. Формирование и вывод результатов ---
    results_df = pd.DataFrame({
        'text': texts, # Сохраняем полный текст
        'tfidf_prediction': tfidf_predictions,
        'linguistic_prediction': linguistic_predictions
    })
    
    output_filename = "classification_results.csv"
    print(f"\nСохранение результатов в файл: {output_filename}...")
    results_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*80)
    print(f"Результаты классификации сохранены в '{output_filename}'.")
    print("================================================================================\n")
    print("Пример первых 5 результатов:")
    print(results_df.head())

if __name__ == "__main__":
    classify_texts()
