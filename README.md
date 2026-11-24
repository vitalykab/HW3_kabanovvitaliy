# RuReviews · классификация тональности

Сравнение TF-IDF с логистической регрессией, ruBERT-tiny2 и Qwen2.5-0.5B-Instruct на фиксированном разбиении RuReviews. Метрика в таблице — macro-F1.

| Модель | Весь test, 8 578 | Общие 150 |
|---|---:|---:|
| Символьный TF-IDF | 0,7320 | 0,7353 |
| ruBERT, эпоха 2 | **0,7412** | 0,7341 |
| Qwen, few-shot | — | 0,5663 |

**Для первого запуска очереди ручной проверки выбран TF-IDF.** При пороге 0,55 он нашёл 1 652 из 2 852 негативных отзывов и дал 410 ложных отправок. Precision — 0,8012, recall — 0,5792. У ruBERT лучший результат для трёхклассовой классификации на полном test.

На полном test обе модели чаще всего путали negative с neutral: 837 случаев у TF-IDF и 961 у ruBERT. На общих 150 отзывах TF-IDF и ruBERT разошлись 26 раз. Qwen вернула 149 валидных JSON из 150.

[Все метрики](results/metrics.csv) · [Три примера ошибок](results/error_examples.csv) · [Семь контрольных ответов](answers.md) · [Результаты и исходные ответы](results/)

### Воспроизведение

Python 3.12, запуск из корня репозитория. Скрипт скачает [RuReviews](https://github.com/sismetanin/rureviews) и обе модели. Проверено на Apple M4 Pro с PyTorch MPS. Код также поддерживает CPU и CUDA.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh
tensorboard --logdir results/tensorboard
```
