"""
Тест кэширования моделей.

Проверяет:
    1. ВСЕ веса скачиваются в стандартный HF-кэш
       (как у transformers/diffusers — так делают senior-разработчики).
    2. При повторной загрузке используется кэш (без повторного скачивания).

Запуск:
    python tests/test_cache.py
"""

import os
import sys
from pathlib import Path

# Добавляем корень проекта в путь
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hybridvoice.config import HybridVoiceConfig
from hybridvoice.vc import SeedVC


def _dir_size_mb(path: Path) -> float:
    """
    Возвращает размер папки в мегабайтах.

    Args:
        path: Путь к папке.

    Returns:
        Размер в МБ (float).
    """
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def test_seedvc_cache_in_hf():
    """
    Проверяет, что SeedVC качает веса в СТАНДАРТНЫЙ HF-кэш.

    Проверяем на маленьком файле CAMPPlus (~27 МБ):
        - 1-й раз: скачивание (если ещё нет в кэше)
        - 2-й раз: загрузка из кэша (HF_HUB_OFFLINE=1)
    """
    print("=" * 60)
    print("ТЕСТ 1: Кэширование SeedVC в стандартный HF-кэш")
    print("=" * 60)

    from huggingface_hub.constants import HF_HUB_CACHE

    config = HybridVoiceConfig()
    vc = SeedVC(config)

    print(f"Фактический кэш SeedVC: {vc.checkpoint_dir}")
    print(f"Стандартный HF-кэш:     {Path(HF_HUB_CACHE)}")
    assert vc.checkpoint_dir == Path(HF_HUB_CACHE), (
        "SeedVC должен качать в стандартный HF-кэш!"
    )

    print("\n[1/2] Первая загрузка CAMPPlus...")
    path1 = vc._load_from_hf("funasr/campplus", "campplus_cn_common.bin")
    print(f"  Файл: {path1}")
    assert Path(path1).exists(), "Файл CAMPPlus не найден!"
    assert str(Path(path1)).startswith(str(Path(HF_HUB_CACHE))), (
        "Файл должен быть внутри HF-кэша!"
    )

    print("\n[2/2] Повторная загрузка в OFFLINE-режиме (только кэш)...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        path2 = vc._load_from_hf("funasr/campplus", "campplus_cn_common.bin")
        assert path1 == path2, "Пути при повторной загрузке должны совпадать!"
        print(f"  ✅ Загружено из кэша (без интернета)")
    finally:
        os.environ["HF_HUB_OFFLINE"] = "0"

    print("\n✅ ТЕСТ 1 ПРОЙДЕН: SeedVC использует стандартный HF-кэш.\n")


def test_custom_cache_dir():
    """
    Проверяет, что config.cache_dir переопределяет папку кэша.
    """
    print("=" * 60)
    print("ТЕСТ 2: Переопределение кэша через config.cache_dir")
    print("=" * 60)

    custom = ROOT / "test_custom_cache"
    config = HybridVoiceConfig(cache_dir=str(custom))
    vc = SeedVC(config)

    print(f"  config.cache_dir = {custom}")
    print(f"  фактический кэш  = {vc.checkpoint_dir}")
    assert vc.checkpoint_dir == custom, "cache_dir должен переопределять кэш!"
    print("  ✅ Переопределение работает.\n")


def test_hf_cache_summary():
    """
    Показывает сводку по всем весам в HF-кэше.
    """
    print("=" * 60)
    print("ТЕСТ 3: Сводка по кэшу весов HybridVoice")
    print("=" * 60)

    from huggingface_hub.constants import HF_HUB_CACHE

    models = {
        "OmniVoice (TTS)": "models--k2-fsa--OmniVoice",
        "Whisper (семантика)": "models--openai--whisper-small",
        "BigVGAN (вокодер)": "models--nvidia--bigvgan_v2_44khz_128band_512x",
        "SeedVC DiT (клонир.)": "models--Plachta--Seed-VC",
        "RMVPE (F0)": "models--lj1995--VoiceConversionWebUI",
        "CAMPPlus (тембр)": "models--funasr--campplus",
    }

    hub = Path(HF_HUB_CACHE)
    total = 0.0
    for label, folder in models.items():
        size = _dir_size_mb(hub / folder)
        status = "✅ в кэше" if size > 0 else "⬜ не скачан"
        print(f"  {label:24} {size:8.1f} МБ  {status}")
        total += size

    print(f"\n  {'ИТОГО':24} {total:8.1f} МБ ({total/1024:.1f} ГБ)")
    print(f"  Расположение: {hub}\n")


if __name__ == "__main__":
    test_seedvc_cache_in_hf()
    test_custom_cache_dir()
    test_hf_cache_summary()
    print("=" * 60)
    print("🎉 ВСЕ ТЕСТЫ КЭШИРОВАНИЯ ПРОЙДЕНЫ!")
    print("=" * 60)
