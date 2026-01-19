import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def create_agreement_heatmap():
    """
    Создает heatmap, показывающий пересечение предсказаний двух моделей.
    """
    try:
        input_file = "classification_results.csv"
        output_file = "model_agreement_heatmap.png"

        print(f"Загрузка файла: {input_file}...")
        df = pd.read_csv(input_file)

        # Получаем уникальные метки, чтобы оси были в правильном порядке
        labels = sorted(np.unique(np.concatenate((df['tfidf_prediction'], df['linguistic_prediction']))))

        print("Построение матрицы пересечений...")
        # Строим confusion matrix между предсказаниями двух моделей
        cm = confusion_matrix(
            y_true=df['linguistic_prediction'], 
            y_pred=df['tfidf_prediction'], 
            labels=labels
        )

        # Создаем DataFrame для более понятных меток на графике
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        print("Создание и сохранение heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Heatmap пересечений предсказаний моделей', fontsize=16)
        plt.xlabel('Предсказания TF-IDF модели', fontsize=12)
        plt.ylabel('Предсказания лингвистической модели', fontsize=12)
        
        # Сохраняем график в файл
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        print("-" * 50)
        print(f"Готово! Heatmap сохранен в файл: {output_file}")
        print("-" * 50)

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Убедитесь, что установлены библиотеки: pip install pandas seaborn matplotlib scikit-learn")

if __name__ == "__main__":
    create_agreement_heatmap()
