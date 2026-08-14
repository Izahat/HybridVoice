"""
Вспомогательные утилиты HybridVoice.

Функции для определения устройства, загрузки и сохранения аудио.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def get_best_device() -> str:
    """
    Автоматически определяет лучшее доступное устройство для инференса.

    Порядок приоритета:
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU

    Returns:
        Строка с именем устройства: "cuda", "mps" или "cpu".

    Example:
        >>> device = get_best_device()
        >>> print(device)
        'cuda'
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Загружает аудиофайл и возвращает waveform + sample rate.

    Поддерживает все форматы, которые понимает soundfile/librosa:
    WAV, MP3, FLAC, OGG, M4A и др.

    Args:
        path: Путь к аудиофайлу.
        target_sr: Если указан — аудио будет ресемплировано
            к этой частоте дискретизации.

    Returns:
        Кортеж (waveform, sample_rate), где waveform — это
        одномерный numpy-массив float32 со значениями в диапазоне [-1, 1].

    Raises:
        FileNotFoundError: Если файл не найден.

    Example:
        >>> audio, sr = load_audio("speech.wav", target_sr=24000)
        >>> print(audio.shape, sr)
        (48000,) 24000
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")

    waveform, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return waveform.astype(np.float32), sr


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int,
) -> Path:
    """
    Сохраняет numpy-массив аудио в файл.

    Автоматически создаёт родительские папки, если их нет.

    Args:
        audio: Одномерный numpy-массив с waveform (float32, диапазон [-1, 1]).
        path: Путь к выходному файлу (например, "output.wav").
        sample_rate: Частота дискретизации.

    Returns:
        Path к сохранённому файлу.

    Example:
        >>> save_audio(audio_array, "result.wav", sample_rate=24000)
        PosixPath('result.wav')
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(path), audio, sample_rate)
    logger.info(f"Аудио сохранено: {path}")
    return path


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Ресемплирует аудио к другой частоте дискретизации.

    Args:
        audio: Входной waveform (numpy-массив).
        orig_sr: Исходная частота дискретизации.
        target_sr: Целевая частота дискретизации.

    Returns:
        Ресемплированный numpy-массив.

    Example:
        >>> audio_24k = resample_audio(audio_16k, orig_sr=16000, target_sr=24000)
    """
    if orig_sr == target_sr:
        return audio

    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Возвращает длительность аудио в секундах.

    Args:
        audio: Numpy-массив с waveform.
        sample_rate: Частота дискретизации.

    Returns:
        Длительность в секундах (float).

    Example:
        >>> duration = get_audio_duration(audio, 24000)
        >>> print(f"Длительность: {duration:.2f} сек")
        Длительность: 3.50 сек
    """
    return len(audio) / sample_rate
