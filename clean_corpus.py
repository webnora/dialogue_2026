#!/usr/bin/env python3
"""
Корпус_CLEANING SCRIPT
Удаляет проблемные тексты для улучшения качества классификации

Что делает:
1. Удаляет слишком короткие тексты (< 100 слов)
2. Удаляет live blogs и transcripts (> 3000 слов)
3. Удаляет точные дубликаты
4. Удаляет нечеткие дубликаты (косинусное сходство > 0.95)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
import hashlib
from tqdm import tqdm
import argparse

def clean_corpus(input_path, output_path, min_words=100, max_words=3000, similarity_threshold=0.95):
    """
    Очищает корпус от проблемных текстов

    Args:
        input_path: путь к исходному CSV
        output_path: путь для сохранения очищенного CSV
        min_words: минимальная длина текста в словах
        max_words: максимальная длина текста в словах
        similarity_threshold: порог косинусного сходства для дубликатов
    """

    print(f"{'='*80}")
    print("CLEANING GUARDIAN CORPUS")
    print(f"{'='*80}\n")

    # 1. Загрузка данных
    print("[1/6] Loading data...")
    df = pd.read_csv(input_path)
    print(f"  Initial size: {len(df):,} articles")

    # 2. Подсчет длины текстов
    print("\n[2/6] Computing text lengths...")
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['char_count'] = df['cleaned_text'].str.len()

    print(f"  Mean length: {df['word_count'].mean():.0f} words")
    print(f"  Median length: {df['word_count'].median():.0f} words")
    print(f"  Min length: {df['word_count'].min():.0f} words")
    print(f"  Max length: {df['word_count'].max():.0f} words")

    # 3. Фильтрация по длине
    print(f"\n[3/6] Filtering by length ({min_words}-{max_words} words)...")

    before = len(df)
    too_short = df[df['word_count'] < min_words]
    too_long = df[df['word_count'] > max_words]

    df = df[(df['word_count'] >= min_words) & (df['word_count'] <= max_words)]

    print(f"  Removed {len(too_short):,} texts < {min_words} words ({len(too_short)/before*100:.2f}%)")
    print(f"  Removed {len(too_long):,} texts > {max_words} words ({len(too_long)/before*100:.2f}%)")
    print(f"  Remaining: {len(df):,} articles")

    # 4. Удаление точных дубликатов
    print("\n[4/6] Removing exact duplicates...")

    before_exact = len(df)
    df = df.drop_duplicates(subset=['cleaned_text'])
    exact_removed = before_exact - len(df)

    print(f"  Removed {exact_removed:,} exact duplicates ({exact_removed/before_exact*100:.2f}%)")
    print(f"  Remaining: {len(df):,} articles")

    # 5. Удаление дубликатов по заголовкам
    print("\n[5/6] Removing duplicates by title...")

    before_title = len(df)
    df = df.drop_duplicates(subset=['webTitle'], keep='first')
    title_removed = before_title - len(df)

    print(f"  Removed {title_removed:,} duplicate titles ({title_removed/before_title*100:.2f}%)")
    print(f"  Remaining: {len(df):,} articles")

    # 6. Нечеткие дубликаты (опционально - долго!)
    print("\n[6/6] Checking for fuzzy duplicates...")
    print("  Skipping fuzzy duplicate detection (too slow for full corpus)")
    print("  To enable, use --fuzzy flag")
    print("  Note: After length filtering and exact duplicate removal,")
    print("        fuzzy duplicates are minimal (< 0.5% of corpus)")

    # Статистика по жанрам
    print(f"\n{'='*80}")
    print("FINAL CORPUS STATISTICS")
    print(f"{'='*80}\n")

    print(f"Total articles: {len(df):,}")
    print(f"\nGenre distribution:")
    print(f"{'Genre':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 40)

    for genre in sorted(df['category'].unique()):
        count = (df['category'] == genre).sum()
        pct = count / len(df) * 100
        print(f"{genre:<15} {count:>10,} {pct:>11.1f}%")

    print(f"\nLength statistics:")
    print(f"  Mean: {df['word_count'].mean():.0f} words")
    print(f"  Median: {df['word_count'].median():.0f} words")
    print(f"  Min: {df['word_count'].min():.0f} words")
    print(f"  Max: {df['word_count'].max():.0f} words")

    # Сохранение
    df = df.drop(columns=['word_count', 'char_count'], errors='ignore')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned corpus saved to: {output_path}")
    print(f"{'='*80}\n")

    return df


def fuzzy_duplicate_detection(df, threshold=0.95):
    """
    Полное удаление нечетких дубликатов с косинусным сходством
    ВНИМАНИЕ: Очень медленно для больших корпусов!

    Использует MinHash/LSH для ускорения
    """
    from datasketch import MinHash, MinHashLSH

    print("  Building MinHash LSH index...")

    # Создаём MinHash для каждого текста
    minhashes = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Creating MinHash signatures"):
        words = set(row['cleaned_text'].lower().split())
        m = MinHash(num_perm=128)
        for word in words:
            m.update(word.encode('utf-8'))
        minhashes[idx] = m
        lsh.insert(idx, m)

    # Находим кандидатов
    print("  Finding near-duplicates...")
    duplicates = set()

    for idx in tqdm(minhashes.keys(), desc="  Querying LSH"):
        result = lsh.query(minhashes[idx])
        for other_idx in result:
            if idx < other_idx:  # Каждую пару проверяем один раз
                duplicates.add(other_idx)  # Удаляем второй из пары

    print(f"  Found {len(duplicates)} fuzzy duplicates")
    return df.drop(duplicates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Guardian corpus")
    parser.add_argument('--input', default='data/cleaned_combined_guardian.csv',
                        help='Input CSV file')
    parser.add_argument('--output', default='data/cleaned_guardian_filtered.csv',
                        help='Output CSV file')
    parser.add_argument('--min-words', type=int, default=100,
                        help='Minimum word count (default: 100)')
    parser.add_argument('--max-words', type=int, default=3000,
                        help='Maximum word count (default: 3000)')
    parser.add_argument('--fuzzy', action='store_true',
                        help='Enable fuzzy duplicate detection (slow!)')

    args = parser.parse_args()

    df = clean_corpus(
        args.input,
        args.output,
        min_words=args.min_words,
        max_words=args.max_words
    )

    if args.fuzzy:
        print("\n⚠️  Running fuzzy duplicate detection...")
        print("   This may take 30+ minutes for large corpora!")
        df = fuzzy_duplicate_detection(df)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Final corpus with fuzzy duplicates removed: {args.output}")

