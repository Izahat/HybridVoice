# HybridVoice 🎙️

Библиотека для генерации речи с клонированием голоса.
Пайплайн как в production-системе **Dublaj**:

- **OmniVoice** — TTS в режиме Voice Design (генерирует речь из текста)
- **SeedVC F0** — Voice Conversion (клонирует голос из reference_audio)

Пользователь работает с **одним объектом** и **одним методом** `.generate()`,
даже не зная, что внутри работают две модели.

## Архитектура пайплайна

```
generate()
   │
   ├─► ШАГ 1: OmniVoice [Voice Design]
   │          текст → речь с голосом ПО УМОЛЧАНИЮ (male/female по языку)
   │
   ├─► ШАГ 2: SeedVC F0 [наложение голоса]
   │          накладывает тембр из reference_audio на речь
   │
   └─► AudioResult (готовое аудио, 44100 Hz)
```

> ⚠️ **Важная логика:** даже при клонировании OmniVoice генерирует речь
> с голосом **по умолчанию** (Voice Design, обычный male/female на нужном
> языке) — **без** референс-аудио. Клонирование делает **SeedVC**: он берёт
> `reference_audio` и накладывает его тембр на речь от OmniVoice.

## Установка

```bash
git clone https://github.com/hybridvoice/HybridVoice.git
cd HybridVoice

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

> ⚠️ Для GPU-ускорения установите PyTorch с CUDA:
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`

## Быстрый старт

```python
from hybridvoice import HybridVoice

model = HybridVoice()

result = model.generate(
    text="Привет, мир!",
    reference_audio="voice.wav",   # голос для клонирования
    language="ru",
)

result.save("output.wav")
```

## Параметры Dublaj (по умолчанию)

### SeedVC V1 F0 Base (44100 Hz)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `diffusion_steps` | 100 | Максимальное качество |
| `length_adjust` | 1.0 | Нормальная скорость |
| `inference_cfg_rate` | 0.8 | Сходство с референсом |
| `auto_f0_adjust` | True | Автоподстройка высоты голоса |
| `f0_condition` | True | F0 модель (всегда) |
| `pitch_shift` | 0 | Без сдвига по полутонам |

- **Чекпоинт**: `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth`
- **Vocoder**: BigVGAN `nvidia/bigvgan_v2_44khz_128band_512x`
- **Pitch extractor**: RMVPE

### OmniVoice (24000 Hz)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `num_steps` | 32 | Шаги диффузии |
| `voice` | "male" | Instruct-голос (по языку) |
| `language` | ISO 639 | Код языка |

## Режимы работы

### 1. Клонирование голоса (основной пайплайн)

Пользователь передаёт: **текст**, **язык** и **аудио для клонирования**.

Логика:
```
1. OmniVoice генерирует речь с DEFAULT-голосом (Voice Design)
   — он НЕ клонирует, просто делает обычный male/female голос на нужном языке
2. SeedVC берёт reference_audio и НАКЛАДЫВАЕТ его тембр на эту речь
```

```python
result = model.generate(
    text="Привет!",
    reference_audio="voice.wav",   # аудио для клонирования
    language="ru",
)
result.save("output.wav")
```

#### 🔧 Параметры SeedVC можно менять на лету

Все параметры SeedVC переопределяются прямо в `generate()` для конкретного
вызова (если не указаны — берутся из config):

```python
result = model.generate(
    text="Привет!",
    reference_audio="voice.wav",
    language="ru",
    # --- Параметры SeedVC (переопределяют config) ---
    diffusion_steps=50,        # меньше = быстрее, больше = качественнее
    inference_cfg_rate=0.9,    # выше = ближе к референсу (0.0-1.0)
    length_adjust=1.0,         # скорость/длина
    auto_f0_adjust=True,       # автоподстройка высоты голоса
    pitch_shift=2,             # сдвиг тона в полутонах
)
```

### 2. Только TTS (без клонирования)
```python
result = model.generate(
    text="Hello!",
    language="en",
    female=True,
    skip_voice_conversion=True,
)
```

### 3. Полный доступ к OmniVoice («точь-в-точь»)
Библиотека даёт доступ ко **ВСЕМ** возможностям OmniVoice через `model.tts`:
```python
# Voice Design с любыми параметрами генерации
audio = model.tts.design("Hello!", language="en", speed=1.2, guidance_scale=2.5)

# Полный pass-through к OmniVoice.generate() (все режимы и параметры)
audio = model.tts.generate(
    text="Hello!",
    instruct="female, young adult, high pitch",
    num_step=32,
    t_shift=0.1,
    class_temperature=0.0,
)

# OmniVoice-клонирование (без SeedVC)
audio = model.tts.clone("Hello!", ref_audio="voice.wav")

# Переиспользуемый voice clone prompt
prompt = model.tts.create_voice_clone_prompt("voice.wav")
prompt.save("my_voice.pt")

# ASR (распознавание речи)
text = model.tts.transcribe("voice.wav")

# Список поддерживаемых языков OmniVoice
ids = model.tts.supported_language_ids()

# Прямой доступ к модели OmniVoice (после загрузки)
omnivoice_model = model.tts.model
```

### 4. Прямой доступ к SeedVC
```python
audio = model.vc.convert(source=tts_audio, target="voice.wav", source_sr=24000)
```

## Маппинг язык → голос

Как в Dublaj: для каждого языка по умолчанию мужской голос,
суффикс `_female` даёт женский.

```python
from hybridvoice import get_voice_for_language, supported_languages

print(supported_languages())   # 18 языков
get_voice_for_language("ru")                  # "male"
get_voice_for_language("ru", female=True)     # "female"
```

## Конфигурация

```python
from hybridvoice import HybridVoice, HybridVoiceConfig

config = HybridVoiceConfig(
    device="cuda",              # "cuda" / "cpu" / "mps" / None (авто)
    diffusion_steps=100,        # SeedVC: макс качество
    inference_cfg_rate=0.8,     # SeedVC: сходство с референсом
    num_steps=32,               # OmniVoice: шаги диффузии
    default_voice="male",       # голос по умолчанию
)

model = HybridVoice(config)
```

## Структура проекта

```
HybridVoice/
├── hybridvoice/
│   ├── __init__.py     # публичный API
│   ├── model.py        # главный класс HybridVoice
│   ├── config.py       # конфигурация (параметры Dublaj)
│   ├── audio.py        # AudioResult (.save, .duration)
│   ├── tts.py          # обёртка OmniVoice (design/clone/generate — полный API)
│   ├── vc.py           # обёртка SeedVC F0 (RMVPE, BigVGAN 44k)
│   ├── voices.py       # маппинг язык → голос
│   └── utils.py        # утилиты (device, аудио)
├── examples/
│   └── example.py      # пример использования
├── checkpoints/        # кэш весов моделей
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Требования

- Python >= 3.10
- PyTorch >= 2.4
- Репозиторий [seed-vc](https://github.com/Plachtaa/Seed-VC) рядом с проектом
  (или задайте путь через переменную окружения `SEED_VC_PATH`)

## Лицензия

Apache-2.0
