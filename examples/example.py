"""
Пример использования библиотеки HybridVoice.

Демонстрирует:
    1. Основной пайплайн: TTS + клонирование голоса через SeedVC
    2. Только TTS (без клонирования)
    3. Полный доступ к OmniVoice напрямую

ВАЖНАЯ ЛОГИКА:
    Даже при клонировании OmniVoice генерирует речь с голосом ПО УМОЛЧАНИЮ
    (male/female по языку). Клонирование делает SeedVC: он берёт
    reference_audio и накладывает его тембр на речь от OmniVoice.

Запуск:
    python examples/example.py
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from hybridvoice import HybridVoice, HybridVoiceConfig, supported_languages


def main():
    """Главная функция примера."""
    # ------------------------------------------------------------------
    # 1. Конфигурация (параметры по умолчанию)
    # ------------------------------------------------------------------
    config = HybridVoiceConfig(
        device=None,              # автоопределение (cuda/mps/cpu)
        diffusion_steps=100,      # SeedVC: макс качество
        inference_cfg_rate=0.8,   # SeedVC: сходство с референсом
        f0_condition=True,        # SeedVC: F0 модель
        num_steps=32,             # OmniVoice: шаги диффузии
        default_voice="male",     # голос по умолчанию
    )

    model = HybridVoice(config)
    print(model)
    print(f"Поддерживаемые языки: {supported_languages()}\n")

    reference_audio = "reference.wav"
    has_ref = Path(reference_audio).exists()

    # ==================================================================
    # ПРИМЕР 1: Основной пайплайн — TTS + клонирование через SeedVC
    # ==================================================================
    # OmniVoice генерирует речь с default-голосом (male для "ru"),
    # затем SeedVC накладывает голос из reference_audio.
    if has_ref:
        print("=" * 60)
        print("ПРИМЕР 1: TTS + клонирование голоса (SeedVC)")
        print("=" * 60)
        result = model.generate(
            text="Привет! Это тест библиотеки HybridVoice.",
            reference_audio=reference_audio,   # голос для клонирования
            language="ru",
            # Параметры SeedVC можно менять прямо здесь (опционально).
            # Если не указать — возьмутся значения из config.
            diffusion_steps=100,       # макс качество
            inference_cfg_rate=0.8,    # сходство с референсом
            auto_f0_adjust=True,       # автоподстройка высоты
            pitch_shift=0,             # без сдвига тона
        )
        result.save("output_cloned.wav")
        print(f"✅ Сохранено: output_cloned.wav ({result.duration:.2f}s)\n")
    else:
        print(f"⚠️  '{reference_audio}' не найден — пропускаю клонирование.\n")

    # ==================================================================
    # ПРИМЕР 2: Только TTS (без клонирования)
    # ==================================================================
    print("=" * 60)
    print("ПРИМЕР 2: Только TTS (Voice Design)")
    print("=" * 60)
    result = model.generate(
        text="Hello! This is OmniVoice.",
        language="en",
        female=True,             # женский голос
        skip_voice_conversion=True,
    )
    result.save("output_tts.wav")
    print(f"✅ Сохранено: output_tts.wav ({result.duration:.2f}s)\n")

    # ==================================================================
    # ПРИМЕР 3: Полный доступ к OmniVoice напрямую
    # ==================================================================
    print("=" * 60)
    print("ПРИМЕР 3: Прямой доступ к OmniVoice (все параметры)")
    print("=" * 60)
    # Voice Design с кастомными параметрами генерации
    audio = model.tts.design(
        "Добрый день!",
        language="ru",
        speed=1.0,
        num_step=32,              # любой параметр OmniVoice
        guidance_scale=2.0,
    )
    print(f"✅ Voice Design: {len(audio)} сэмплов\n")

    # OmniVoice-клонирование (без SeedVC) — если есть референс
    if has_ref:
        audio = model.tts.clone("Hello!", ref_audio=reference_audio)
        print(f"✅ OmniVoice Clone: {len(audio)} сэмплов\n")

    # ------------------------------------------------------------------
    # Освобождаем память
    # ------------------------------------------------------------------
    model.unload()
    print("Готово!")


if __name__ == "__main__":
    main()
