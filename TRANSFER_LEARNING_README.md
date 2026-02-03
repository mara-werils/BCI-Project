# BCI Transfer Learning & HER2 Classification Pipeline

Расширенная версия BCI репозитория с поддержкой **Transfer Learning** и **HER2 классификации**.

## 📋 Содержание

1. [Обзор](#обзор)
2. [Структура проекта](#структура-проекта)
3. [Установка](#установка)
4. [Быстрый старт](#быстрый-старт)
5. [Детальное руководство](#детальное-руководство)
6. [Результаты](#результаты)
7. [Для статьи](#для-статьи)

---

## 🎯 Обзор

### Задачи:
1. **Transfer Learning Pipeline** для обучения на малых данных (10-50% от полного датасета)
2. **HER2 Classification** - классификация молекулярных подтипов (0, 1+, 2+, 3+)

### Архитектура:
```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING PIPELINE               │
└─────────────────────────────────────────────────────────────┘

Phase 1: Pre-training (Full Dataset: 3,896 pairs)
H&E Image → [PyramidPix2pix Generator] → IHC Image
                    ↓
           [PatchGAN Discriminator]
           Loss: L1 + L2 + L3 + L4 (Pyramid) + GAN

Phase 2: Fine-tuning (Small Dataset: 10-50%)
Pre-trained Generator → [Fine-tune] → Adapted Generator
                    ↓
           [Strong Augmentation]
           - Color Jitter
           - Affine Transform
           - Gaussian Noise

Phase 3: Classification (Optional)
[Generator Encoder] → [Classification Head] → HER2 Class
                              ↓
                    [0, 1+, 2+, 3+]
```

---

## 📁 Структура проекта

```
BCI-main/
├── BCI_dataset/                    # Полный датасет
│   ├── HE/
│   │   ├── train/                  # 3,896 изображений H&E
│   │   └── test/                   # 977 изображений H&E
│   └── IHC/
│       ├── train/                  # 3,896 изображений IHC
│       └── test/                   # 977 изображений IHC
│
├── BCI_dataset_small_10pct/        # 10% от датасета (создается скриптом)
├── BCI_dataset_small_20pct/        # 20% от датасета
├── BCI_dataset_small_50pct/        # 50% от датасета
│
├── PyramidPix2pix/                 # Основной код модели
│   ├── data/
│   │   ├── aligned_dataset.py
│   │   └── her2_aligned_dataset.py # [NEW] Dataset с HER2 метками
│   ├── models/
│   │   ├── pix2pix_model.py
│   │   └── pix2pix_transfer_model.py # [NEW] Transfer + Classification
│   ├── options/
│   │   └── train_options.py        # [MODIFIED] Новые аргументы
│   ├── util/
│   │   └── her2_utils.py           # [NEW] Утилиты для HER2
│   ├── train.py
│   └── test.py
│
├── scripts/                        # [NEW] Скрипты для экспериментов
│   ├── create_small_dataset.py     # Создание подмножеств
│   └── run_experiments.py          # Запуск всех экспериментов
│
├── notebooks/                      # [NEW] Jupyter ноутбуки
│   ├── analysis.ipynb              # Анализ результатов
│   └── figures/                    # Графики и визуализации
│
├── experiments/                    # [NEW] Результаты экспериментов
│   ├── checkpoints/                # Сохраненные модели
│   ├── results/                    # Результаты тестирования
│   └── logs/                       # Логи обучения
│
└── TRANSFER_LEARNING_README.md     # Эта документация
```

---

## 🔧 Установка

### 1. Требования
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (для GPU)
- ~12GB VRAM (для RTX 3060)

### 2. Установка зависимостей

```bash
cd BCI-main/PyramidPix2pix
pip install -r requirements.txt

# Дополнительные библиотеки для анализа
pip install scikit-image scikit-learn seaborn jupyter
```

### 3. Проверка GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🚀 Быстрый старт

### Вариант 1: Запуск всех экспериментов

```bash
cd BCI-main
python scripts/run_experiments.py --experiment all --gpu_ids 0
```

Это выполнит:
1. Создание малых датасетов (10%, 20%, 50%)
2. Pre-training на полном датасете
3. Fine-tuning на малых датасетах
4. Обучение с HER2 классификатором
5. Оценку всех моделей

### Вариант 2: Пошаговый запуск

```bash
# Шаг 1: Создать малые датасеты
python scripts/create_small_dataset.py --all

# Шаг 2: Pre-training (полный датасет)
python scripts/run_experiments.py --experiment pretrain

# Шаг 3: Fine-tuning (10% данных)
python scripts/run_experiments.py --experiment finetune --ratio 0.1

# Шаг 4: Обучение с классификатором
python scripts/run_experiments.py --experiment classify
```

---

## 📖 Детальное руководство

### 1. Создание малых датасетов

```bash
# Создать 10% подмножество
python scripts/create_small_dataset.py \
    --source ./BCI_dataset \
    --target ./BCI_dataset_small_10pct \
    --ratio 0.1 \
    --seed 42

# Или создать все сразу
python scripts/create_small_dataset.py --all
```

**Важно**: Скрипт сохраняет распределение HER2 статусов (stratified sampling).

### 2. Pre-training на полном датасете

```bash
cd PyramidPix2pix

python train.py \
    --dataroot ../datasets/BCI \
    --name bci_pretrain_full \
    --gpu_ids 0 \
    --pattern L1_L2_L3_L4 \
    --batch_size 2 \
    --crop_size 512 \
    --preprocess crop \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --save_epoch_freq 10
```

### 3. Fine-tuning с Transfer Learning

```bash
python train.py \
    --dataroot ../datasets/BCI_small_10pct \
    --name bci_finetune_10pct \
    --gpu_ids 0 \
    --pattern L1_L2_L3_L4 \
    --pretrained_path ../experiments/checkpoints/bci_pretrain_full/latest_net_G.pth \
    --strong_augment \
    --finetune_lr_factor 0.1 \
    --batch_size 4 \
    --n_epochs 30 \
    --n_epochs_decay 20
```

**Новые аргументы:**
- `--pretrained_path`: путь к pre-trained модели
- `--freeze_encoder`: заморозить encoder (опционально)
- `--strong_augment`: усиленные аугментации
- `--finetune_lr_factor`: множитель для LR (0.1 = в 10 раз меньше)

### 4. Обучение с HER2 классификацией

```bash
python train.py \
    --dataroot ../datasets/BCI \
    --name bci_with_classifier \
    --model pix2pix_transfer \
    --dataset_mode her2_aligned \
    --gpu_ids 0 \
    --pattern L1_L2_L3_L4 \
    --enable_classification \
    --lambda_classifier 0.5 \
    --num_classes 4 \
    --n_epochs 50 \
    --n_epochs_decay 50
```

**Новые аргументы:**
- `--enable_classification`: включить HER2 классификатор
- `--lambda_classifier`: вес classification loss
- `--num_classes`: количество классов (4 для HER2)
- `--class_weighted_loss`: взвешенный loss для несбалансированных данных

### 5. Тестирование

```bash
python test.py \
    --dataroot ../datasets/BCI \
    --name bci_pretrain_full \
    --gpu_ids 0 \
    --preprocess none
```

### 6. Оценка метрик

```bash
python evaluate.py --result_path ./results/bci_pretrain_full
```

---

## 📊 Результаты

### Ожидаемые метрики

| Method | Data | PSNR ↑ | SSIM ↑ | Training Time |
|--------|------|--------|--------|---------------|
| Baseline (Full) | 3,896 | 21.16 | 0.477 | ~12h |
| Transfer 50% | 1,948 | ~20.5 | ~0.46 | ~6h |
| Transfer 20% | 780 | ~19.5 | ~0.43 | ~2.5h |
| Transfer 10% | 390 | ~18.0 | ~0.38 | ~1h |
| No Transfer 10% | 390 | ~15.5 | ~0.30 | ~1h |

### HER2 Classification

| Method | Accuracy | Notes |
|--------|----------|-------|
| Multi-task (Full) | ~65-75% | Generation + Classification |
| ResNet-18 (Real IHC) | ~70-80% | Baseline on real images |

---

## 📝 Для статьи

### Jupyter Notebook

Откройте `notebooks/analysis.ipynb` для:
- Визуализации распределения данных
- Расчета метрик (PSNR, SSIM)
- Построения confusion matrix
- Генерации таблиц в LaTeX

### Methods Section Draft

См. файл `notebooks/methods_draft.md` с черновиком раздела Methods.

### Примеры визуализации

После запуска экспериментов графики сохраняются в:
- `notebooks/figures/dataset_distribution.png`
- `notebooks/figures/confusion_matrix.png`
- `notebooks/figures/visualization_*.png`

---

## 🔬 Технические детали

### Архитектура генератора
- **ResNet-9blocks** (по умолчанию)
- Input: 3 канала (RGB)
- Output: 3 канала (RGB)
- 9 ResNet блоков в bottleneck

### Архитектура классификатора
```python
HER2ClassificationHead(
    input_channels=256,
    num_classes=4,
    dropout=0.5
)
# Structure: AdaptiveAvgPool → 256 → 512 → 256 → 4
```

### Аугментации для малых данных
- Color Jitter: brightness=0.3, contrast=0.3, saturation=0.2
- Affine: rotation=±10°, translate=±5%, scale=0.9-1.1
- Gaussian Blur: kernel=3-5, sigma=0.1-2.0
- Gaussian Noise: std=0.01-0.05

---

## ❓ FAQ

**Q: Сколько времени занимает обучение?**
- Full dataset (100 epochs): ~12-15 часов на RTX 3060
- Small dataset (50 epochs): ~1-3 часа

**Q: Можно ли использовать CPU?**
- Да, но очень медленно. Добавьте `--gpu_ids -1`

**Q: Как уменьшить использование памяти?**
- Уменьшите `--batch_size` до 1
- Уменьшите `--crop_size` до 256

**Q: Как продолжить прерванное обучение?**
```bash
python train.py ... --continue_train --epoch latest
```

---

## 📚 Цитирование

```bibtex
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Shengjie and Zhu, Chuang and Xu, Feng and Jia, Xinyu and Shi, Zhongyue and Jin, Mulan},
    title     = {BCI: Breast Cancer Immunohistochemical Image Generation Through Pyramid Pix2pix},
    booktitle = {CVPR Workshops},
    year      = {2022},
    pages     = {1815-1824}
}
```

---

## 📧 Контакты

При возникновении вопросов обращайтесь к оригинальным авторам BCI:
- Shengjie Liu (shengjie.Liu@bupt.edu.cn)
- Chuang Zhu (czhu@bupt.edu.cn)


