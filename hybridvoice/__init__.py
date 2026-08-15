"""
HybridVoice — библиотека для генерации речи с клонированием голоса.

Пайплайн:
    - OmniVoice (TTS, режим Voice Design) — генерирует речь из текста
    - SeedVC F0 (Voice Conversion) — клонирует голос из reference_audio

Быстрый старт:
    >>> from hybridvoice import HybridVoice
    >>> model = HybridVoice()
    >>> result = model.generate(
    ...     text="Привет!",
    ...     reference_audio="voice.wav",
    ...     language="ru",
    ... )
    >>> result.save("output.wav")
"""

from .audio import AudioResult
from .config import HybridVoiceConfig
from .model import HybridVoice
from .tts import OmniVoiceTTS
from .vc import SeedVC
from .voices import LANGUAGE_VOICE_MAP, all_languages, get_voice_for_language, list_languages, supported_languages

__version__ = "0.2.0"

__all__ = [
    "HybridVoice",
    "HybridVoiceConfig",
    "AudioResult",
    "OmniVoiceTTS",
    "SeedVC",
    "LANGUAGE_VOICE_MAP",
    "get_voice_for_language",
    "supported_languages",
    "all_languages",
    "list_languages",
    "__version__",
]
