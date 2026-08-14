"""
Модуль TTS на базе OmniVoice.

Предоставляет ПОЛНЫЙ доступ ко всем возможностям OmniVoice:
    1. Voice Design — голос описывается через instruct ("male"/"female"...).
    2. Auto Voice — модель сама выбирает голос (без instruct и ref_audio).
    3. Voice Clone — клонирование голоса средствами OmniVoice.
    + Все параметры генерации (guidance_scale, speed, температуры и т.д.).

В основном пайплайне HybridVoice для клонирования используется НЕ
OmniVoice-клонирование, а SeedVC: OmniVoice генерирует речь с голосом
по умолчанию (Voice Design), а SeedVC накладывает целевой голос.
Метод clone() оставлен для тех, кому нужно именно OmniVoice-клонирование.
"""

import logging
from typing import List, Optional, Union

import numpy as np

from .config import HybridVoiceConfig
from .utils import get_best_device
from .voices import get_voice_for_language

logger = logging.getLogger(__name__)


class OmniVoiceTTS:
    """
    Обёртка для TTS-модели OmniVoice с полным доступом к её API.

    Attributes:
        config: Конфигурация HybridVoice.
        device: Устройство, на котором работает модель.
        model: Загруженная модель OmniVoice (None до первого вызова).
            После загрузки даёт прямой доступ ко всем методам OmniVoice:
            generate(), create_voice_clone_prompt(), transcribe() и т.д.

    Example:
        >>> tts = OmniVoiceTTS(config)
        >>> # Voice Design (голос по языку)
        >>> audio = tts.design(text="Hello!", language="en")
        >>> # Полный контроль (все параметры OmniVoice)
        >>> audio = tts.generate(text="Hello!", instruct="female", speed=1.2)
        >>> # OmniVoice-клонирование
        >>> audio = tts.clone(text="Hello!", ref_audio="voice.wav")
    """

    def __init__(self, config: HybridVoiceConfig):
        """
        Инициализирует обёртку TTS (модель ещё НЕ загружается).

        Args:
            config: Конфигурация HybridVoice с параметрами TTS.
        """
        self.config = config
        self.device = config.device or get_best_device()
        self.model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """
        Загружена ли модель.

        Returns:
            True, если модель уже загружена в память.
        """
        return self._is_loaded

    @property
    def sampling_rate(self) -> int:
        """
        Частота дискретизации модели OmniVoice (обычно 24000 Hz).

        Returns:
            Sample rate модели.
        """
        if self.model is not None and hasattr(self.model, "sampling_rate"):
            return self.model.sampling_rate
        return self.config.tts_sample_rate

    def load(self) -> "OmniVoiceTTS":
        """
        Загружает модель OmniVoice в память.

        Вызывается автоматически при первом вызове generate(),
        но можно вызвать вручную для предзагрузки.

        Returns:
            self (для цепочки вызовов).

        Raises:
            ImportError: Если пакет omnivoice не установлен.

        Example:
            >>> tts = OmniVoiceTTS(config).load()
        """
        if self._is_loaded:
            logger.info("Модель TTS уже загружена, пропускаем.")
            return self

        logger.info(f"Загрузка OmniVoice из '{self.config.tts_model}'...")
        logger.info(f"Устройство: {self.device}")

        try:
            from omnivoice.models.omnivoice import OmniVoice
        except ImportError as e:
            raise ImportError(
                "Пакет 'omnivoice' не найден. "
                "Установите его: pip install -e ./OmniVoice"
            ) from e

        dtype = self.config.get_torch_dtype()

        self.model = OmniVoice.from_pretrained(
            self.config.tts_model,
            device_map=self.device,
            dtype=dtype,
        )

        self._is_loaded = True
        logger.info(f"OmniVoice загружена. Sample rate: {self.sampling_rate}")
        return self

    # ============================================================
    # ГЛАВНЫЙ МЕТОД: полный доступ к OmniVoice.generate()
    # ============================================================
    def generate(
        self,
        text: Union[str, List[str]],
        language: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_audio: Optional[Union[str, tuple]] = None,
        instruct: Optional[str] = None,
        duration: Optional[float] = None,
        speed: Optional[float] = None,
        return_all: bool = False,
        **gen_kwargs,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Полный pass-through к OmniVoice.generate() — все режимы и параметры.

        Это главный метод для доступа ко ВСЕМ возможностям OmniVoice:
        Voice Design, Auto Voice, Voice Clone, батч-генерация и любые
        параметры генерации.

        Args:
            text: Текст (строка или список строк для батча).
            language: Язык ("en", "ru", "zh"...). None = language-agnostic.
            ref_text: Транскрипт референса (для режима Voice Clone).
            ref_audio: Референс-аудио (для режима Voice Clone).
                Путь к файлу или кортеж (waveform, sample_rate).
            instruct: Описание голоса (для режима Voice Design),
                например "female, young adult, high pitch".
            duration: Фиксированная длительность в секундах. None = авто.
            speed: Скорость речи (>1.0 быстрее, <1.0 медленнее).
            return_all: Если True — вернуть список всех аудио (для батча).
                Если False — вернуть только первое аудио.
            **gen_kwargs: ЛЮБЫЕ параметры генерации OmniVoice:
                num_step (int): число шагов диффузии (по умолчанию 32).
                guidance_scale (float): сила guidance (по умолчанию 2.0).
                t_shift (float): сдвиг таймстепов (по умолчанию 0.1).
                denoise (bool): токен denoise (по умолчанию True).
                layer_penalty_factor (float): по умолчанию 5.0.
                position_temperature (float): по умолчанию 5.0.
                class_temperature (float): 0 = greedy (по умолчанию 0.0).
                postprocess_output (bool): постобработка (по умолчанию True).
                audio_chunk_duration (float): чанки для длинных (15.0).
                и другие (см. OmniVoiceGenerationConfig).

        Returns:
            Numpy-массив с waveform, или список массивов если return_all=True.

        Example:
            >>> # Voice Design со всеми параметрами
            >>> audio = tts.generate(
            ...     text="Hello!", instruct="female",
            ...     num_step=32, guidance_scale=2.0, speed=1.1,
            ... )
            >>> # Батч-генерация
            >>> audios = tts.generate(["Hi", "Bye"], return_all=True)
        """
        if not self._is_loaded:
            self.load()

        # Параметры из config как дефолты, gen_kwargs их переопределяют
        gen_kwargs.setdefault("num_step", self.config.num_steps)

        logger.info(f"OmniVoice.generate(): режим авто-определён по аргументам")

        audios: List[np.ndarray] = self.model.generate(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio=ref_audio,
            instruct=instruct,
            duration=duration,
            speed=speed,
            **gen_kwargs,
        )

        if return_all:
            return audios
        return audios[0]

    # ============================================================
    # Режим Voice Design — удобный метод
    # ============================================================
    def design(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        female: bool = False,
        duration: Optional[float] = None,
        speed: Optional[float] = None,
        **gen_kwargs,
    ) -> np.ndarray:
        """
        Синтез в режиме Voice Design (без референс-аудио).

        Голос задаётся через instruct. Если voice=None — подбирается
        по языку через маппинг (male по умолчанию). Это режим, который
        используется в основном пайплайне HybridVoice перед SeedVC.

        Args:
            text: Текст для синтеза.
            voice: Instruct-голос ("male", "female", ...).
                Если None — подбирается по языку.
            language: Код языка ISO 639.
            female: Если True и voice=None — женский голос.
            duration: Фиксированная длительность (сек). None = авто.
            speed: Скорость речи.
            **gen_kwargs: Доп. параметры генерации OmniVoice.

        Returns:
            Numpy-массив с waveform.

        Example:
            >>> audio = tts.design("Привет", language="ru")          # male
            >>> audio = tts.design("Привет", language="ru", female=True)
        """
        _language = language if language is not None else self.config.language

        if voice is None:
            voice = get_voice_for_language(
                _language,
                female=female,
                default_voice=self.config.default_voice,
            )

        logger.info(f"TTS [Voice Design]: '{text[:40]}...' голос='{voice}'")

        return self.generate(
            text=text,
            language=_language,
            instruct=voice,
            duration=duration,
            speed=speed,
            **gen_kwargs,
        )

    # ============================================================
    # Режим Voice Clone (средствами OmniVoice)
    # ============================================================
    def clone(
        self,
        text: str,
        ref_audio: Union[str, tuple],
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        speed: Optional[float] = None,
        **gen_kwargs,
    ) -> np.ndarray:
        """
        Клонирование голоса средствами OmniVoice (с референс-аудио).

        ВНИМАНИЕ: в основном пайплайне HybridVoice клонирование делает
        SeedVC, а не этот метод. Этот метод — для тех, кому нужно
        именно OmniVoice-клонирование без SeedVC.

        Args:
            text: Текст для синтеза.
            ref_audio: Референс-аудио (путь или (waveform, sr)).
            ref_text: Транскрипт референса (опционально).
            language: Целевой язык (для cross-lingual).
            duration: Фиксированная длительность.
            speed: Скорость речи.
            **gen_kwargs: Доп. параметры генерации OmniVoice.

        Returns:
            Numpy-массив с waveform.

        Example:
            >>> audio = tts.clone("Hello!", ref_audio="voice.wav")
        """
        _language = language if language is not None else self.config.language

        logger.info(f"TTS [OmniVoice Clone]: '{text[:40]}...'")

        # По умолчанию максимальное качество для клонирования
        gen_kwargs.setdefault("num_step", 32)

        return self.generate(
            text=text,
            language=_language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            duration=duration,
            speed=speed,
            **gen_kwargs,
        )

    def unload(self) -> None:
        """
        Выгружает модель из памяти и освобождает VRAM.

        Example:
            >>> tts.unload()
        """
        if self.model is not None:
            import torch

            del self.model
            self.model = None
            self._is_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Модель TTS выгружена из памяти.")

    # ============================================================
    # PASS-THROUGH методы OmniVoice (доступ «точь-в-точь»)
    # ============================================================

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ):
        """
        Создаёт переиспользуемый voice clone prompt из референс-аудио.

        Pass-through к OmniVoice.create_voice_clone_prompt().
        Полезно, если один и тот же референс используется много раз —
        prompt создаётся один раз и передаётся в generate().

        Args:
            ref_audio: Путь к файлу или кортеж (waveform, sample_rate).
            ref_text: Транскрипт референса. Если None — авто-транскрипция
                через ASR (модель Whisper загрузится автоматически).
            preprocess_prompt: Применять удаление тишины и обрезку.

        Returns:
            Объект VoiceClonePrompt (можно сохранить через .save()).

        Example:
            >>> prompt = tts.create_voice_clone_prompt("voice.wav")
            >>> prompt.save("my_voice.pt")   # сохранить на будущее
        """
        if not self._is_loaded:
            self.load()
        return self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )

    def load_asr_model(self, model_name: Optional[str] = None) -> None:
        """
        Загружает Whisper ASR модель для авто-транскрипции референсов.

        Pass-through к OmniVoice.load_asr_model().

        Args:
            model_name: Имя Whisper-модели. По умолчанию
                openai/whisper-large-v3-turbo.

        Example:
            >>> tts.load_asr_model()
            >>> text = tts.transcribe("voice.wav")
        """
        if not self._is_loaded:
            self.load()
        self.model.load_asr_model(model_name)

    def transcribe(self, audio: Union[str, tuple]) -> str:
        """
        Распознаёт речь из аудио (ASR через Whisper).

        Pass-through к OmniVoice.transcribe(). Требует загруженной
        ASR-модели (загрузится автоматически при необходимости).

        Args:
            audio: Путь к файлу или кортеж (waveform, sample_rate).

        Returns:
            Распознанный текст.

        Example:
            >>> text = tts.transcribe("voice.wav")
        """
        if not self._is_loaded:
            self.load()
        return self.model.transcribe(audio)

    def supported_language_ids(self) -> set:
        """
        Возвращает множество поддерживаемых кодов языков OmniVoice.

        Pass-through к OmniVoice.supported_language_ids().

        Returns:
            Множество кодов языков (например, {"en", "ru", "zh", ...}).

        Example:
            >>> ids = tts.supported_language_ids()
        """
        if not self._is_loaded:
            self.load()
        return self.model.supported_language_ids()

    def supported_language_names(self) -> set:
        """
        Возвращает множество поддерживаемых названий языков OmniVoice.

        Pass-through к OmniVoice.supported_language_names().

        Returns:
            Множество названий языков (например, {"English", "Russian", ...}).

        Example:
            >>> names = tts.supported_language_names()
        """
        if not self._is_loaded:
            self.load()
        return self.model.supported_language_names()
