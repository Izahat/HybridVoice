"""
Тест кэширования моделей.

Проверяет:
    1. Файлы моделей скачиваются в нужные папки кэша.
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
from hybridvoice.vc import SeedVC, _find_seed_vc_repo


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


def _list_files(path: Path) -> list:
    """
    Возвращает список файлов в папке (рекурсивно).

    Args:
        path: Путь к папке.

    Returns:
        Список относительных путей к файлам.
    """
    if not path.exists():
        return []
    return sorted(
        str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()
    )


def test_seedvc_cache():
    """
    Проверяет, что SeedVC скачивает веса в checkpoints/ и использует кэш.

    Скачивает маленький файл CAMPPlus (~7 МБ) дважды:
        - 1-й раз: скачивание с HuggingFace
        - 2-й раз: загрузка из кэша (проверяем через HF_HUB_OFFLINE)
    """
    print("=" * 60)
    print("ТЕСТ 1: Кэширование SeedVC")
    print("=" * 60)

    config = HybridVoiceConfig()
    vc = SeedVC(config)

    print(f"Папка кэша SeedVC: {vc.checkpoint_dir}")
    assert vc.checkpoint_dir.is_absolute(), "Путь к кэшу должен быть абсолютным!"

    # --- Скачиваем маленький CAMPPlus для проверки механизма ---
    print("\n[1/3] Первое скачивание CAMPPlus (~7 МБ)...")
    files_before = _list_files(vc.checkpoint_dir)
    path1 = vc._load_from_hf("funasr/campplus", "campplus_cn_common.bin")
    files_after = _list_files(vc.checkpoint_dir)

    new_files = set(files_after) - set(files_before)
    print(f"  Скачано в кэш: {new_files}")
    print(f"  Путь к файлу: {path1}")
    assert Path(path1).exists(), "Файл CAMPPlus не найден после скачивания!"
    assert len(new_files) > 0, "В кэше не появилось новых файлов!"

    size_mb = Path(path1).stat().st_size / (1024 * 1024)
    print(f"  Размер: {size_mb:.1f} МБ")

    # --- Проверяем, что повторная загрузка идёт из кэша ---
    print("\n[2/3] Повторная загрузка в OFFLINE-режиме (только кэш)...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        path2 = vc._load_from_hf("funasr/campplus", "campplus_cn_common.bin")
        assert path1 == path2, "Пути при повторной загрузке должны совпадать!"
        print(f"  ✅ Загружено из кэша (без интернета): {path2}")
    finally:
        os.environ["HF_HUB_OFFLINE"] = "0"

    # --- Итог по папке кэша ---
    print(f"\n[3/3] Содержимое кэша SeedVC:")
    print(f"  Папка: {vc.checkpoint_dir}")
    print(f"  Размер: {_dir_size_mb(vc.checkpoint_dir):.1f} МБ")
    for f in _list_files(vc.checkpoint_dir)[:10]:
        print(f"    - {f}")

    print("\n✅ ТЕСТ 1 ПРОЙДЕН: SeedVC кэширует веса и использует кэш.\n")


def test_omnivoice_cache_location():
    """
    Показывает, куда OmniVoice скачивает веса (стандартный HF-кэш).

    Сам OmniVoice НЕ скачивает здесь (это ~3-5 ГБ) — только показывает
    расположение кэша. Реальное скачивание происходит в test_pipeline.py.
    """
    print("=" * 60)
    print("ТЕСТ 2: Расположение кэша OmniVoice")
    print("=" * 60)

    from huggingface_hub.constants import HF_HUB_CACHE

    print(f"  Стандартный HF-кэш: {HF_HUB_CACHE}")
    print(f"  OmniVoice будет скачан в:")
    print(f"    {HF_HUB_CACHE}/models--k2-fsa--OmniVoice/")
    print(f"  Audio-tokenizer в:")
    print(f"    {HF_HUB_CACHE}/models--eustlb--higgs-audio-v2-tokenizer/")
    print("\n✅ ТЕСТ 2 ПРОЙДЕН: расположение кэша OmniVoice определено.\n")


def test_seedvc_repo_found():
    """
    Проверяет, что репозиторий seed-vc найден на диске.
    """
    print("=" * 60)
    print("ТЕСТ 3: Репозиторий seed-vc")
    print("=" * 60)

    repo = _find_seed_vc_repo()
    print(f"  Репозиторий: {repo}")
    assert repo is not None, "Репозиторий seed-vc не найден!"
    assert (repo / "modules").exists(), "Папка modules/ не найдена в seed-vc!"
    print(f"  ✅ Репозиторий и modules/ найдены.\n")


if __name__ == "__main__":
    test_seedvc_repo_found()
    test_omnivoice_cache_location()
    test_seedvc_cache()
    print("=" * 60)
    print("🎉 ВСЕ ТЕСТЫ КЭШИРОВАНИЯ ПРОЙДЕНЫ!")
    print("=" * 60)
