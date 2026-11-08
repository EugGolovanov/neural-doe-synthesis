# Сохранённое предобучение на 27 000 четвёрках

Исходный запуск: `mixed_ssim_27k_3ep_20261107T100359_750664Z`.
Это архив фактически использованных исходников и результатов, не повторное обучение и не переписанная реализация метода.

## Веса

- Итоговые веса после трёх эпох: [`../inputs/network_phase/checkpoint_003.pt`](../inputs/network_phase/checkpoint_003.pt).
- Checkpoint после первой эпохи: [`run/checkpoint_001.pt`](run/checkpoint_001.pt).

Оба файла сохраняют исходный формат: `model`, `momenta`, `config`, номер эпохи и состояния генераторов случайных чисел. Итоговый checkpoint используется `synthesize_phase.py` напрямую. Его копия побайтово сверена с исходным файлом завершённого запуска. Загружать следует только доверенные `.pt`.

## Исходники

Точка входа: [`source/experiments/experiment_unet_physics_adapter/pretrain_mixed.py`](source/experiments/experiment_unet_physics_adapter/pretrain_mixed.py).

`source/` — точная копия сохранённого в момент запуска снимка, включая связанные модули физики, модели, оптимизатора и подготовки данных, документацию и исходный `working_tree.diff`. Внутренние имена пакетов `experiments` сохранены намеренно для воспроизводимости; корневая архивная папка проекта `experiments/` для запуска не нужна.

## Настройки

27 000 обучающих четвёрок = 108 000 изображений, по 36 000 из MNIST, Fashion-MNIST и CIFAR-10. Validation и test — по 99 четвёрок. Идентификаторы исходных изображений, контрольные суммы подготовленных PNG и точный порядок четвёрок сохранены в `run/inputs.json.gz`.

Три эпохи, seed=17, seed данных и перестановок=2026. U-Net width=8, rank=2, alpha=2, два внутренних GS+Muon шага. Внешний Muon обучает веса сети; это не последующая прямая оптимизация фазы. Обучающий loss: maximum-normalized intensity MSE + 0.5×(1−SSIM). Внутренний loss адаптера — исходный MSE, с полными производными через развёртку.

Фактические настройки: `run/parameters.json`; среда и исходная ревизия: `run/environment.json`. История: `run/history.jsonl`, порядок примеров по эпохам: `run/epoch_*_order.json`, оценки: `run/validation_*.json`, `run/test_metrics.json`, статус исходного запуска: `run/status.json`.

## Повторный запуск

Исторический runner требует Apple MPS, Python 3.12 и достаточно памяти для microbatch=256. Автоматической смены устройства или размера batch нет. Запуск дорогой; сохранение этого архива его не выполняет.

Из корня репозитория:

```bash
python3 -m pip install torch==2.10.0 torchvision==0.25.0 numpy Pillow
cd pretraining_27k/source
python3 -m experiments.experiment_unet_physics_adapter.pretrain_mixed
```

Важно запускать из указанной папки: тогда импортируется именно сохранённая версия кода. Официальные датасеты скачиваются torchvision, подготовленные PNG создаются заново. Их массивы и исходные архивы не включены в этот комплект. Для сопоставления повторного запуска с историческим используются сохранённые идентификаторы и хеши; одинаковый seed не гарантирует побитового совпадения на другой платформе или с другими версиями библиотек.

Результаты повторного запуска появятся в `source/experiments/experiment_unet_physics_adapter/results/`, без перезаписи сохранённого `run/`.

`run/inputs.json.gz` — gzip-сжатая без изменения содержимого исходная JSON-запись:

```python
import gzip
import json

with gzip.open('pretraining_27k/run/inputs.json.gz', 'rt') as stream:
    manifest = json.load(stream)
```

Хеши всех сохранённых файлов и несжатого manifest перечислены в `manifest.json`.
