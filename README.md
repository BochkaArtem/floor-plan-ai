# Floor Plan AI

Веб-прототип для автоматической разметки архитектурных планировок и условной
генерации новых планировок.

Кейс «Генерация архитектурных планов этажа с помощью нейросети»:

* загрузка изображения планировки → автоматическое распознавание стен, окон,
  дверей и комнат;
* canvas-редактор для ручной коррекции полигонов;
* экспорт разметки в формате **COCO/JSON** (и обратный импорт);
* генерация новых планировок по условиям (площадь, количество комнат, форма
  границы) — на выбор: процедурный или нейросетевой (pix2pix-UNet) бэкенд;
* готовые скрипты для обучения сегментационной (UNet) и генеративной
  (pix2pix) моделей на синтетике или **CubiCasa5K**.

> Из коробки приложение работает без обучения: для сегментации используется
> классический CV-пайплайн на OpenCV, для генерации — процедурный генератор.
> Если положить веса в `models/segmentation.pt` и `models/generator.pt`,
> backend автоматически переключится на нейросетевые модели.

## Скриншоты

Демонстрационные сгенерированные планировки лежат в `demo/generated/`,
размеченная демо-выборка — в `demo/images/` и `demo/annotations/demo.coco.json`.

## Стек

* **Backend**: Python 3.10+, Flask, OpenCV, scikit-image, PyTorch
* **Frontend**: vanilla JS, [Konva.js](https://konvajs.org/) для canvas-редактора
* **Аннотации**: COCO (object detection schema, polygon segmentation)
* **Модели**: компактные UNet и pix2pix-UNet (см. `backend/ml/`)

## Структура репозитория

```
floor-plan-ai/
├── backend/
│   ├── app.py              Flask entrypoint
│   ├── routes/             /api/upload, /api/detect, /api/export, /api/generate
│   ├── ml/
│   │   ├── segmentation.py UNet + classical CV fallback
│   │   ├── unet.py         compact UNet model
│   │   └── generation.py   procedural + pix2pix generators
│   ├── utils/
│   │   ├── coco.py         COCO build / load helpers
│   │   ├── synth.py        процедурный генератор планировок
│   │   └── image.py        I/O утилиты, общая палитра классов
│   └── static/             frontend (HTML/CSS/JS)
├── scripts/
│   ├── make_synthetic_demo.py        генерирует демо-выборку (12 планировок + COCO)
│   ├── make_generated_examples.py    рендерит примеры для demo/generated/
│   ├── download_cubicasa5k.py        скачивает CubiCasa5K (5 ГБ)
│   ├── train_segmentation.py         обучает UNet
│   └── train_generator.py            обучает pix2pix-UNet
├── tests/                  pytest тесты
├── demo/                   демо-выборка и примеры генерации (включены в репо)
├── data/                   датасеты (gitignored)
├── models/                 веса моделей (gitignored)
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Установка

Требуется Python 3.10+.

```bash
git clone https://github.com/BochkaArtem/floor-plan-ai.git
cd floor-plan-ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Запуск

```bash
python -m backend.app
```

Откройте http://localhost:8000.

* Вкладка **«Разметка»** — загрузите изображение → «Авто-разметка» →
  при необходимости поправьте полигоны вручную → «Скачать COCO JSON».
* Вкладка **«Генерация»** — задайте площадь / количество комнат / форму
  границы → «Сгенерировать план». Кнопка «Перенести в редактор разметки»
  отправит результат во вкладку разметки для дальнейшего редактирования.

Эндпоинт `/health` возвращает текущий backend (`classical-cv` vs `unet`,
`procedural` vs `pix2pix`).

## API

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/api/upload` | multipart `image=…` или JSON `{ "data_url": "data:image/…" }`; возвращает `image_id` |
| `POST` | `/api/detect` | `{ "image_id": "…" }` → массив полигонов по категориям |
| `POST` | `/api/export/coco` | `{ images: [{ file_name, width, height, polygons:[{category, points}] }] }` → COCO JSON |
| `POST` | `/api/import/coco` | multipart `file=…` (COCO JSON) → массив изображений с полигонами |
| `POST` | `/api/generate` | `{ width, height, num_rooms, area_m2?, boundary_shape?, seed? }` → PNG (data URL) + полигоны |
| `GET`  | `/health` | состояние моделей |

## Демо-выборка и сгенерированные примеры

```bash
# Регенерировать demo/images/ + demo/annotations/demo.coco.json
python scripts/make_synthetic_demo.py --num 12

# Регенерировать demo/generated/*.png (6 примеров)
python scripts/make_generated_examples.py
```

## Обучение моделей

### Сегментация (UNet)

Smoke-обучение на синтетике (CPU, 1–2 минуты на эпоху):

```bash
python scripts/train_segmentation.py \
    --coco demo/annotations/demo.coco.json \
    --images-dir demo/images \
    --epochs 10 \
    --out models/segmentation.pt
```

Полноценное обучение на CubiCasa5K (нужен GPU):

```bash
python scripts/download_cubicasa5k.py --out data/cubicasa5k

# Конвертируйте CubiCasa5K в COCO (см. CubiCasa5K README) либо адаптируйте
# FloorPlanDataset под структуру CubiCasa5K и затем:
python scripts/train_segmentation.py \
    --coco data/cubicasa5k_coco.json \
    --images-dir data/cubicasa5k/images \
    --epochs 50 --batch-size 16 --image-size 512
```

После обучения backend автоматически загрузит `models/segmentation.pt` и
переключится с `classical-cv` на `unet`.

### Генерация (pix2pix-UNet)

```bash
python scripts/train_generator.py \
    --num-samples 1024 \
    --epochs 30 \
    --batch-size 8 \
    --out models/generator.pt
```

Веса по умолчанию ищутся в `models/generator.pt`; при наличии backend
переключится с `procedural` на `pix2pix`.

## Тесты и линтер

```bash
pytest -q
ruff check .
```

## Идеи для развития

* Заменить кодировщик UNet на pretrained ResNet-18 (через
  `segmentation_models_pytorch`) — заметный прирост качества.
* Добавить графовый conditional GAN (HouseGAN-стиль) для генерации с явным
  bubble-diagram'ом.
* Векторизация результата через `floortrans` или OpenCV (Hough + skeletonize)
  для экспорта в DXF/IFC.
* Обучение сегментации на полном CubiCasa5K с аугментациями (rotate, scale,
  illumination) — дает выраженный прирост на реальных сканированных планах.
* Ручные операции с инсоляцией и оконными проёмами (метрики: освещённость,
  ориентация по сторонам света).

## Лицензия

MIT — см. `LICENSE` (если требуется добавьте по необходимости).
