import pandas as pd
import joblib
import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

# --- Копируем код из предыдущих шагов ---
warnings.filterwarnings('ignore')
STANCE_MARKERS = ['arguably', 'reportedly', 'seemingly', 'apparently', 'undoubtedly', 'clearly', 'obviously', 'evidently', 'supposedly', 'presumably', 'ostensibly']
HEDGES = ['perhaps', 'possibly', 'somewhat', 'rather', 'quite', 'relatively', 'comparatively']
MODAL_VERBS = ['can', 'could', 'may', 'might', 'must', 'should', 'ought', 'would', 'shall']
REPORTING_VERBS = ['said', 'says', 'say', 'told', 'tells', 'tell', 'claimed', 'claims', 'claim', 'stated', 'states', 'state', 'reported', 'reports', 'report', 'announced', 'announces']

def extract_linguistic_features(text, nlp):
    doc = nlp(text)
    features = {}
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    features['type_token_ratio'] = len(set(tokens)) / len(tokens) if tokens else 0
    sentences = list(doc.sents)
    features['avg_sentence_length'] = np.mean([len(s) for s in sentences]) if sentences else 0
    total_tokens = len(doc)
    if total_tokens > 0:
        features['first_person_ratio'] = sum(1 for t in doc if t.tag_ == 'PRP' and t.text.lower() in ['i', 'we', 'me', 'us', 'my', 'our']) / total_tokens
        features['second_person_ratio'] = sum(1 for t in doc if t.tag_ == 'PRP' and t.text.lower() in ['you', 'your']) / total_tokens
        features['third_person_ratio'] = sum(1 for t in doc if t.tag_ == 'PRP' and t.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their']) / total_tokens
        features['modal_ratio'] = sum(1 for t in doc if t.text.lower() in MODAL_VERBS) / total_tokens
        features['stance_markers_ratio'] = sum(1 for t in doc if t.text.lower() in STANCE_MARKERS) / total_tokens
        features['hedges_ratio'] = sum(1 for t in doc if t.text.lower() in HEDGES) / total_tokens
        features['quotes_ratio'] = text.count('"') / total_tokens
        features['reporting_verbs_ratio'] = sum(1 for t in doc if t.text.lower() in REPORTING_VERBS) / total_tokens
    else:
        for key in ['first_person_ratio', 'second_person_ratio', 'third_person_ratio', 'modal_ratio', 'stance_markers_ratio', 'hedges_ratio', 'quotes_ratio', 'reporting_verbs_ratio']:
            features[key] = 0
    return features
# --- Конец скопированного кода ---

def create_scatter_plots():
    """
    Создает 2D scatter plot визуализации для TF-IDF и лингвистических моделей.
    """
    try:
        print("Загрузка данных...")
        results_df = pd.read_csv("classification_results.csv")
        texts = results_df['text'].tolist()

        # --- 1. Визуализация для TF-IDF ---
        print("\n[1/2] Создание визуализации для TF-IDF модели...")
        print("  - Загрузка векторизатора и трансформация текстов...")
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        tfidf_vectors = tfidf_vectorizer.transform(texts)

        print("  - Применение t-SNE (может занять время)...")
        tsne_tfidf = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tfidf_2d = tsne_tfidf.fit_transform(tfidf_vectors.toarray())

        plot_df_tfidf = pd.DataFrame(tfidf_2d, columns=['x', 'y'])
        plot_df_tfidf['prediction'] = results_df['tfidf_prediction']

        print("  - Отрисовка и сохранение графика...")
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=plot_df_tfidf, x='x', y='y', hue='prediction', palette='deep', s=50)
        plt.title('Визуализация текстов (t-SNE) на основе TF-IDF признаков', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Предсказанный жанр')
        plt.tight_layout()
        plt.savefig('tfidf_scatter_plot.png', dpi=300)
        plt.clf()
        print("  ✓ График 'tfidf_scatter_plot.png' сохранен.")

        # --- 2. Визуализация для лингвистической модели ---
        print("\n[2/2] Создание визуализации для лингвистической модели...")
        print("  - Извлечение лингвистических признаков...")
        nlp = spacy.load('en_core_web_sm')
        linguistic_features_list = [extract_linguistic_features(text, nlp) for text in texts]
        features_df = pd.DataFrame(linguistic_features_list)

        print("  - Применение t-SNE...")
        tsne_ling = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        ling_2d = tsne_ling.fit_transform(features_df)
        
        plot_df_ling = pd.DataFrame(ling_2d, columns=['x', 'y'])
        plot_df_ling['prediction'] = results_df['linguistic_prediction']

        print("  - Отрисовка и сохранение графика...")
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=plot_df_ling, x='x', y='y', hue='prediction', palette='deep', s=50)
        plt.title('Визуализация текстов (t-SNE) на основе лингвистических признаков', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Предсказанный жанр')
        plt.tight_layout()
        plt.savefig('linguistic_scatter_plot.png', dpi=300)
        print("  ✓ График 'linguistic_scatter_plot.png' сохранен.")
        
        print("\n" + "="*50)
        print("Готово! Обе визуализации успешно созданы.")
        print("="*50)

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    create_scatter_plots()
