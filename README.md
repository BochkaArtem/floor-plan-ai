# Floor Plan AI

Веб-прототип для автоматической разметки архитектурных планировок и условной
генерации новых планировок **с учётом семантики комнат** и
**произвольной формы квартиры** (прямоугольник / L / T / U / +
крест / случайный полиамино).

Кейс «Генерация архитектурных планов этажа с помощью нейросети»:

* загрузка изображения планировки → автоматическое распознавание стен,
  окон, дверей и комнат;
* canvas-редактор для ручной коррекции полигонов;
* экспорт разметки в формате **COCO/JSON** (и обратный импорт);
  каждая комната дополнительно несёт `subcategory` с типом
  (`hall`, `living`, `kitchen`, `bedroom`, `bathroom`, `balcony`);
* генерация новых планировок по условиям (площадь, количество комнат,
  форма границы, требуемые типы комнат) — на выбор:
  * **процедурный планировщик** (semantic layout planner) — детерминирован,
    использует архитектурные эвристики (вход → прихожая → гостиная;
    кухня и балкон у наружных стен; санузел рядом со спальней,
    дальше от входа; гостиная — самая большая, центральная);
  * **нейросетевой генератор** (`MaskUNet`) — компактный условный U-Net
    (~488K параметров), обученный на синтетике, генерирует
    5-классовую семантическую маску
    `{background, wall, window, door, room}` →
    редактируемые полигоны для дальнейшей правки;
* готовые скрипты для синтеза датасета и обучения **обеих** моделей
  (сегментационной UNet и генеративной MaskUNet) на CPU за ~25 минут.

> **Готовые веса включены в репо** (Git LFS): `models/segmentation.pt`
> и `models/generator.pt` — после `git clone` бэкенд автоматически
> переключается с `classical-cv` / `procedural` на `unet` /
> `mask-unet`. Чтобы клонировать с весами, нужно установить Git LFS:
> `git lfs install` (один раз) → `git clone …`.

## Скриншоты

Демонстрационные сгенерированные планировки лежат в `demo/generated/`
(процедурный бэкенд, чистая визуализация с подписями комнат),
`demo/generated_nn/` (выход нейросети) и `demo/comparison/`
(side-by-side). Размеченная демо-выборка — в `demo/images/` и
`demo/annotations/demo.coco.json`. Формы и комнаты варьируются:

| Форма | Пример | Комнаты |
| --- | --- | --- |
| Прямоугольник | `01_rect_classic_3rooms.png` | hall, living, kitchen, bedroom |
| L | `02_L_shape_5rooms.png` | hall, living, kitchen, bedroom, bathroom |
| T | `03_T_shape_family.png` | hall, living, kitchen, 2× bedroom, bathroom |
| U | `04_U_shape_with_balcony.png` | hall, living, kitchen, 2× bedroom, bathroom, balcony |
| Крест (+) | `05_plus_cross_5rooms.png` | hall, living, kitchen, bedroom, bathroom |
| Случайная | `06_random_polyomino.png` | hall, living, kitchen, bedroom, bathroom |

## Стек

* **Backend**: Python 3.10+, Flask, OpenCV, scikit-image, PyTorch
* **Frontend**: vanilla JS, [Konva.js](https://konvajs.org/) для canvas-редактора
* **Аннотации**: COCO (object detection schema, polygon segmentation,
  расширение `subcategory` для типа комнаты)
* **Модели**:
  * `backend.ml.unet.UNet` — компактный U-Net для сегментации,
  * `backend.ml.generation.MaskUNet` — условный U-Net для генерации,
  * `backend.ml.generation.Pix2PixUNet` — legacy pix2pix RGB-генератор.

## Структура репозитория

```
floor-plan-ai/
├── backend/
│   ├── app.py              Flask entrypoint
│   ├── routes/             /api/upload, /api/detect, /api/export, /api/generate
│   ├── ml/
│   │   ├── segmentation.py UNet + classical CV fallback (5-классовая разметка)
│   │   ├── unet.py         compact UNet
│   │   └── generation.py   procedural / MaskUNet / Pix2Pix backends
│   ├── utils/
│   │   ├── coco.py         COCO build / load helpers (включая subcategory)
│   │   ├── layout_planner.py  семантический планировщик: формы + типы комнат
│   │   ├── synth.py        рендер планов в RGB / mask / полигоны
│   │   └── image.py        I/O утилиты, общая палитра
│   └── static/             frontend (HTML/CSS/JS)
├── scripts/
│   ├── make_synthetic_demo.py        демо-выборка (12 планировок + COCO)
│   ├── make_generated_examples.py    8 примеров для demo/generated[_nn]/
│   ├── make_training_data.py         синтез датасета (image+mask+conditions)
│   ├── train_models.py               обучает UNet + MaskUNet за один запуск
│   ├── train_segmentation.py         legacy: только UNet на размеченной COCO
│   ├── train_generator.py            legacy: только pix2pix RGB
│   └── download_cubicasa5k.py        скачивает CubiCasa5K (5 ГБ)
├── tests/                  pytest тесты (планировщик, COCO, API)
├── demo/                   демо-выборка и примеры генерации (включены в репо)
├── data/                   датасеты (gitignored)
├── models/                 веса моделей (Git LFS, включены в репо)
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Установка

Требуется Python 3.10+ и [Git LFS](https://git-lfs.com/) (для скачивания
весов моделей).

```bash
# Один раз на машину:
git lfs install

git clone https://github.com/BochkaArtem/floor-plan-ai.git
cd floor-plan-ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Если Git LFS не установлен — клонирование пройдёт, но в `models/*.pt`
лежат текстовые pointer-файлы. Бэкенд это распознает и автоматически
упадёт на `classical-cv` / `procedural`. Чтобы получить веса позже:

```bash
git lfs install
git lfs pull
```

## Запуск

```bash
python -m backend.app
```

Откройте http://localhost:8000.

* Вкладка **«Разметка»** — загрузите изображение → «Авто-разметка» →
  при необходимости поправьте полигоны вручную → «Скачать COCO JSON».
* Вкладка **«Генерация»** — задайте параметры:
  * **форма границы**: прямоугольник, L, T, U, крест, случайная или
    «авто» (рандомная);
  * **типы комнат**: чек-боксы (прихожая, гостиная, кухня, спальня,
    санузел, балкон);
  * **бэкенд**: «авто», «нейросеть» или «правила»;
  * **площадь** / **количество комнат** / **seed**.
  Кнопка «Перенести в редактор разметки» отправит результат во вкладку
  разметки с уже подгруженными редактируемыми полигонами.

Эндпоинт `/health` возвращает текущие бэкенды и список доступных
форм/типов:

```json
{
  "status": "ok",
  "segmenter": "unet",
  "generator": "mask-unet",
  "generator_backends_available": { "procedural": true, "nn": true },
  "boundary_shapes": ["rect", "L", "T", "U", "plus", "random", "auto"],
  "room_types": ["hall", "living", "kitchen", "bedroom", "bathroom", "balcony"]
}
```

## API

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/api/upload` | multipart `image=…` или JSON `{ "data_url": "data:image/…" }`; возвращает `image_id` |
| `POST` | `/api/detect` | `{ "image_id": "…" }` → массив полигонов по категориям |
| `POST` | `/api/export/coco` | `{ images: [{ file_name, width, height, polygons:[{category, subcategory?, points}] }] }` → COCO JSON |
| `POST` | `/api/import/coco` | multipart `file=…` (COCO JSON) → массив изображений с полигонами |
| `POST` | `/api/generate` | `{ width, height, num_rooms, area_m2?, boundary_shape?, room_types?: [...], backend?: "auto"\|"nn"\|"procedural", seed? }` → PNG (data URL) + полигоны (с `subcategory` для комнат) |
| `GET`  | `/health` | состояние моделей и список поддерживаемых форм/типов |

## Демо-выборка и сгенерированные примеры

```bash
# Регенерировать demo/images/ + demo/annotations/demo.coco.json (12 планов)
python scripts/make_synthetic_demo.py --num 12

# Регенерировать demo/generated/ (8 примеров через процедурный планировщик)
python -m scripts.make_generated_examples --backend procedural --out-dir demo/generated

# Регенерировать demo/generated_nn/ (8 примеров через нейросеть)
python -m scripts.make_generated_examples --backend nn --out-dir demo/generated_nn
```

## Обучение моделей

### Быстрый путь — обе нейросети на синтетике (CPU, ~25 мин)

```bash
# 1. Сгенерировать 1500 синтетических планов с тройкой (image, mask, conditions).
python -m scripts.make_training_data --out data/synthetic --n 1500 \
    --width 128 --height 128 --seed 2026

# 2. Обучить UNet (3-канальный вход → 5 классов) и MaskUNet
#    (5-канальный вход = boundary + room_count + 3 типа → 5 классов).
python -m scripts.train_models --data data/synthetic --epochs 12 \
    --batch-size 16 --lr 2e-3 \
    --out-seg models/segmentation.pt --out-gen models/generator.pt
```

После обучения бэкенд автоматически переключится:

* `segmenter` с `classical-cv` → `unet`;
* `generator` с `procedural` → `mask-unet`.

Условия для нейросетевой генерации формируются как 5-канальный тензор:

| Канал | Смысл |
| --- | --- |
| 0 | бинарная маска квартиры (1 — внутри, 0 — снаружи) |
| 1 | равномерное поле `num_rooms / 8` |
| 2-4 | равномерные поля для типов `hall`, `living`, `kitchen` |

Сетка предсказывает 5-классовый softmax (background, wall, window, door,
room) → клампится по маске квартиры → переводится в полигоны через
`cv2.findContours`. Это и есть «нейросеть, которая учится размечать»:
обучающие маски и предсказываемые маски используют один и тот же формат.

Текущие включённые в репо веса обучены на 12 эпохах × 1500 синтетических
примерах и достигают `val_acc=1.000` для сегментации и `val_acc=0.96`
для генерации. Для production-качества стоит дообучить на CubiCasa5K
(см. ниже).

### Сегментация на CubiCasa5K (нужен GPU)

```bash
python scripts/download_cubicasa5k.py --out data/cubicasa5k
# Конвертируйте CubiCasa5K в COCO (см. CubiCasa5K README) либо адаптируйте
# FloorPlanDataset под структуру CubiCasa5K и затем:
python scripts/train_segmentation.py \
    --coco data/cubicasa5k_coco.json \
    --images-dir data/cubicasa5k/images \
    --epochs 50 --batch-size 16 --image-size 512
```

### Legacy: pix2pix RGB-генератор

```bash
python scripts/train_generator.py \
    --num-samples 1024 \
    --epochs 30 \
    --batch-size 8 \
    --out models/generator_pix2pix.pt
```

## Семантика комнат

Планировщик `backend.utils.layout_planner` использует следующую цепочку
эвристик при назначении типов комнат:

1. **Прихожая** размещается у входа (нижний край квартиры по умолчанию).
2. **Гостиная** — самая большая комната, по возможности с наружной стеной
   и рядом с прихожей.
3. **Кухня** — средняя по размеру комната с обязательной наружной стеной
   (для окна / вытяжки).
4. **Спальни** — приватные комнаты, дальше от входа.
5. **Санузел** — компактные комнаты рядом со спальней, не у входа.
6. **Балкон** — назначается комнатам с большим периметром наружной стены,
   если запрошен.

Двери ставятся между парами смежных комнат (общая стена ≥ 30 px) +
входная дверь от прихожей наружу. Окна — на наружных стенах комнат.

## Тесты и линтер

```bash
pytest -q
ruff check .
```

## Идеи для развития

* Заменить кодировщик UNet на pretrained ResNet-18 (через
  `segmentation_models_pytorch`).
* Добавить графовый conditional GAN (HouseGAN-стиль) с явным
  bubble-diagram'ом и более богатыми условиями (соседство, площадь,
  ориентация по сторонам света).
* Векторизация результата через `floortrans` или OpenCV
  (Hough + skeletonize) для экспорта в DXF/IFC.
* Обучение сегментации на полном CubiCasa5K с аугментациями
  (rotate, scale, illumination).
* Расширить семантику: учёт инсоляции, окон по сторонам света,
  нормативные требования (минимальные площади).

## Лицензия

MIT — см. `LICENSE` (если требуется добавьте по необходимости).
