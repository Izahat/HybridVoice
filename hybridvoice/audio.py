"""
Результат генерации аудио.

Класс AudioResult — это обёртка вокруг сгенерированного аудио,
которая предоставляет удобные методы: .save(), .to_numpy(), .duration.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np


class AudioResult:
    """
    Результат генерации аудио от HybridVoice.

    Хранит waveform и sample_rate, предоставляет методы
    для сохранения и доступа к данным.

    Attributes:
        audio: Numpy-массив с waveform (float32, диапазон [-1, 1]).
        sample_rate: Частота дискретизации (обычно 22050 или 24000).
        metadata: Словарь с дополнительной информацией о генерации
            (какие модели использовались, время и т.д.).

    Example:
        >>> result = model.generate(text="Hello!", reference_audio="voice.wav")
        >>> result.save("output.wav")
        >>> print(result.duration)
        3.2
        >>> print(result.sample_rate)
        22050
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[dict] = None,
    ):
        """
        Инициализирует AudioResult.

        Args:
            audio: Numpy-массив с waveform.
            sample_rate: Частота дискретизации.
            metadata: Необязательный словарь с метаданными.
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.metadata = metadata or {}

    @property
    def duration(self) -> float:
        """
        Длительность аудио в секундах.

        Returns:
            Длительность в секундах (float).

        Example:
            >>> print(result.duration)
            3.2
        """
        return len(self.audio) / self.sample_rate

    @property
    def waveform(self) -> np.ndarray:
        """
        Алиас для self.audio — waveform как numpy-массив.

        Returns:
            Numpy-массив с waveform.
        """
        return self.audio

    def to_numpy(self) -> np.ndarray:
        """
        Возвращает waveform как numpy-массив.

        Returns:
            Копия numpy-массива с аудио.

        Example:
            >>> array = result.to_numpy()
            >>> print(array.shape)
            (70560,)
        """
        return self.audio.copy()

    def save(self, path: Union[str, Path]) -> Path:
        """
        Сохраняет аудио в файл.

        Формат определяется расширением файла (.wav, .flac, .mp3 и др.).
        Автоматически создаёт папки, если их нет.

        Args:
            path: Путь к выходному файлу.

        Returns:
            Path к сохранённому файлу.

        Example:
            >>> result.save("output.wav")
            PosixPath('output.wav')
        """
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), self.audio, self.sample_rate)
        return path

    def __repr__(self) -> str:
        """
        Строковое представление объекта.

        Returns:
            Строка с информацией о длительности и sample rate.
        """
        return (
            f"AudioResult(duration={self.duration:.2f}s, "
            f"sample_rate={self.sample_rate}, "
            f"samples={len(self.audio)})"
        )

    def __len__(self) -> int:
        """
        Возвращает количество сэмплов в аудио.

        Returns:
            Число сэмплов (int).
        """
        return len(self.audio)
