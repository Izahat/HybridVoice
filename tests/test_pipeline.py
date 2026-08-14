"""
Тест полного пайплайна HybridVoice (end-to-end).

При первом запуске скачивает все модели в кэш (несколько ГБ),
затем генерирует аудио.

Режимы:
    python tests/test_pipeline.py --tts-only   # только OmniVoice TTS
    python tests/test_pipeline.py --clone      # полный пайплайн (нужен reference.wav)

Запуск:
    python tests/test_pipeline.py --tts-only
    python tests/test_pipeline.py --clone
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from hybridvoice import HybridVoice, HybridVoiceConfig


def test_tts_only():
    """
    Тестирует только OmniVoice TTS (без клонирования).

    Скачивает OmniVoice (~3-5 ГБ при первом запуске) и генерирует речь.
    """
    print("=" * 60)
    print("ТЕСТ: OmniVoice TTS (Voice Design)")
    print("=" * 60)

    config = HybridVoiceConfig(
        num_steps=16,   # меньше шагов для быстрого теста
    )
    model = HybridVoice(config)

    print("\nЗагрузка OmniVoice (при первом запуске скачается)...")
    start = time.time()
    result = model.generate(
        text="Hello! This is a test of HybridVoice.",
        language="en",
        female=True,
        skip_voice_conversion=True,   # только TTS
    )
    elapsed = time.time() - start

    output = ROOT / "test_output_tts.wav"
    result.save(output)

    print(f"\n✅ TTS работает!")
    print(f"   Выход: {output}")
    print(f"   Длительность: {result.duration:.2f} сек")
    print(f"   Sample rate: {result.sample_rate}")
    print(f"   Время генерации: {elapsed:.2f} сек")

    model.unload()


def test_clone():
    """
    Тестирует полный пайплайн: OmniVoice TTS + SeedVC клонирование.

    Требует reference.wav в корне проекта (аудио-образец голоса).
    """
    print("=" * 60)
    print("ТЕСТ: Полный пайплайн (TTS + клонирование SeedVC)")
    print("=" * 60)

    reference = ROOT / "reference.wav"
    if not reference.exists():
        print(f"\n❌ Файл '{reference}' не найден!")
        print("   Положите аудио-образец голоса (3-10 сек речи) в корень проекта")
        print("   и назовите его reference.wav, затем запустите тест снова.")
        return

    config = HybridVoiceConfig(
        num_steps=16,         # TTS: быстрее для теста
        diffusion_steps=20,   # SeedVC: быстрее для теста (в проде 100)
    )
    model = HybridVoice(config)

    print("\nЗапуск полного пайплайна (модели скачаются при первом запуске)...")
    start = time.time()
    result = model.generate(
        text="Привет! Это тест клонирования голоса.",
        reference_audio=str(reference),
        language="ru",
    )
    elapsed = time.time() - start

    output = ROOT / "test_output_cloned.wav"
    result.save(output)

    print(f"\n✅ Полный пайплайн работает!")
    print(f"   Выход: {output}")
    print(f"   Длительность: {result.duration:.2f} сек")
    print(f"   Sample rate: {result.sample_rate} (SeedVC даёт 44100)")
    print(f"   Время генерации: {elapsed:.2f} сек")

    model.unload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тест пайплайна HybridVoice")
    parser.add_argument("--tts-only", action="store_true", help="Только OmniVoice TTS")
    parser.add_argument("--clone", action="store_true", help="Полный пайплайн с клонированием")
    args = parser.parse_args()

    if args.tts_only:
        test_tts_only()
    elif args.clone:
        test_clone()
    else:
        print("Укажите режим:")
        print("  python tests/test_pipeline.py --tts-only")
        print("  python tests/test_pipeline.py --clone")
