"""
Вспомогательные утилиты HybridVoice.

Функции для определения устройства, загрузки и сохранения аудио.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_DEVICE_TYPES = {"cpu", "cuda", "mps"}


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


def resolve_device(device: Optional[str] = None) -> str:
    """Проверяет и возвращает фактическое устройство для инференса.

    Поддерживаются ``cpu``, ``cuda``, ``cuda:N`` и ``mps``. Для явно
    указанного ускорителя дополнительно проверяется его доступность.
    """
    import torch

    candidate = get_best_device() if device is None else str(device).lower().strip()
    if not candidate:
        raise ValueError("device не может быть пустой строкой.")

    try:
        parsed = torch.device(candidate)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Некорректное устройство '{device}'. "
            "Допустимые значения: cpu, cuda, cuda:N, mps."
        ) from exc

    if parsed.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"Устройство '{candidate}' не поддерживается HybridVoice. "
            "Допустимые типы: cpu, cuda, mps."
        )

    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Запрошено устройство CUDA, но torch.cuda.is_available() == False."
            )
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA-устройство cuda:{parsed.index} не существует. "
                f"Доступно устройств: {torch.cuda.device_count()}."
            )

    if parsed.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if not mps_available:
            raise RuntimeError(
                "Запрошено устройство MPS, но MPS недоступно в текущем PyTorch."
            )

    return str(parsed)


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


def validate_reference_audio(
    path: Union[str, Path],
    min_duration: float = 0.5,
    max_duration: float = 25.0,
    silence_threshold: float = 1e-5,
) -> Path:
    """Проверяет reference-аудио до загрузки ML-моделей.

    Проверяется существование файла, декодирование, длительность, конечность
    сэмплов и наличие ненулевого звукового сигнала.
    """
    if min_duration <= 0:
        raise ValueError("min_duration должен быть больше 0.")
    if max_duration < min_duration:
        raise ValueError("max_duration должен быть не меньше min_duration.")
    if silence_threshold < 0:
        raise ValueError("silence_threshold не может быть отрицательным.")

    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Reference-аудио не найдено: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Reference-аудио должно быть файлом: {audio_path}")

    import librosa

    try:
        # Ограничение защищает от загрузки случайно переданного многочасового файла.
        waveform, sample_rate = librosa.load(
            str(audio_path),
            sr=16000,
            mono=True,
            duration=max_duration + 0.1,
        )
    except Exception as exc:
        raise ValueError(
            f"Не удалось декодировать reference-аудио '{audio_path}': {exc}"
        ) from exc

    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError(f"Reference-аудио пустое: {audio_path}")
    if not np.isfinite(waveform).all():
        raise ValueError(
            f"Reference-аудио содержит NaN или бесконечные значения: {audio_path}"
        )

    duration = waveform.size / sample_rate
    if duration < min_duration:
        raise ValueError(
            f"Reference-аудио слишком короткое: {duration:.2f} сек. "
            f"Минимум: {min_duration:.2f} сек."
        )
    if duration > max_duration:
        raise ValueError(
            f"Reference-аудио слишком длинное: больше {max_duration:.2f} сек. "
            "Обрежьте его до участка с чистой речью."
        )

    peak = float(np.max(np.abs(waveform)))
    if peak <= silence_threshold:
        raise ValueError(
            f"Reference-аудио не содержит слышимого сигнала: {audio_path}"
        )

    return audio_path


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
