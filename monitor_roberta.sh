#!/bin/bash
# Монитор прогресса задачи 1.3 (RoBERTa training)

echo "🔄 Проверка статуса задачи 1.3 (RoBERTa)..."
echo ""

# Проверяем, работает ли процесс
if ps aux | grep -v grep | grep "run_roberta_training.py" > /dev/null; then
    echo "✅ Скрипт работает"
    echo ""

    # Проверяем последние 5 строк вывода
    echo "Последние обновления:"
    tail -5 /private/tmp/claude/-Users-nora-src-dialogue_2026/tasks/*.output 2>/dev/null | grep -E "(Epoch|Training|Validating|Accuracy)" || echo "Вывод недоступен"
    echo ""

    # Проверяем наличие результатов
    if [ -f "results/roberta_metrics.json" ]; then
        echo "🎉 ЗАДАЧА ЗАВЕРШЕНА!"
        echo ""
        echo "Результаты:"
        python3 -c "import json; m=json.load(open('results/roberta_metrics.json')); print(f\"Accuracy: {m['test_accuracy']*100:.2f}%\nF1 (macro): {m['test_f1_macro']:.4f}\")"
        exit 0
    else
        echo "⏳ Ещё не завершено..."
        echo ""
        echo "Ориентировочное время выполнения:"
        echo "  - На MPS (Apple Silicon): ~2-3 часа"
        echo "  -   На CPU: ~6-8 часов"
    fi
else
    if [ -f "results/roberta_metrics.json" ]; then
        echo "🎉 ЗАДАЧА ЗАВЕРШЕНА!"
        echo ""
        echo "Результаты:"
        python3 -c "import json; m=json.load(open('results/roberta_metrics.json')); print(f\"Accuracy: {m['test_accuracy']*100:.2f}%\nF1 (macro): {m['test_f1_macro']:.4f}\")"
    else
        echo "⚠️  Скрипт не работает и результаты не найдены"
        echo "Возможно, он был остановлен или завершился с ошибкой"
    fi
fi
