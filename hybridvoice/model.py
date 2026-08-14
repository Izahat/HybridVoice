"""
Главный класс HybridVoice.

Объединяет TTS (OmniVoice) и Voice Conversion (SeedVC F0)
в единый простой интерфейс — как в production-пайплайне Dublaj.

Логика пайплайна:
    1. OmniVoice генерирует речь из текста в режиме Voice Design
       (голос задаётся через instruct: "male"/"female" по языку).
    2. SeedVC преобразует эту речь к тембру reference_audio
       (клонирование голоса делает именно SeedVC, не OmniVoice).

Пользователь работает с одним объектом и одним методом .generate().
"""

import logging
import time
from typing import Optional

from .audio import AudioResult
from .config import HybridVoiceConfig
from .tts import OmniVoiceTTS
from .vc import SeedVC

logger = logging.getLogger(__name__)


class HybridVoice:
    """
    Главная модель HybridVoice: TTS + Voice Conversion в одном API.

    Пайплайн (как в Dublaj):
        Текст → OmniVoice [Voice Design, default голос] → SeedVC [наложение голоса] → результат

    ВАЖНАЯ ЛОГИКА:
        Даже при клонировании OmniVoice генерирует речь с голосом ПО
        УМОЛЧАНИЮ (Voice Design). Клонирование делает SeedVC: он берёт
        reference_audio и накладывает его тембр на речь от OmniVoice.

    ПОЛНЫЙ ДОСТУП К OMNIVOICE:
        Через атрибут `model.tts` доступны ВСЕ возможности OmniVoice:
            - model.tts.design(...)  — Voice Design (голос по языку)
            - model.tts.clone(...)   — OmniVoice-клонирование (без SeedVC)
            - model.tts.generate(...) — полный pass-through всех параметров
            - model.tts.model        — сама модель OmniVoice (после загрузки)

        Через атрибут `model.vc` доступен SeedVC:
            - model.vc.convert(source, target, ...) — конверсия голоса

    Attributes:
        config: Конфигурация пайплайна.
        tts: Обёртка OmniVoice TTS (полный доступ к её API).
        vc: Обёртка SeedVC Voice Conversion.

    Example:
        >>> from hybridvoice import HybridVoice
        >>> model = HybridVoice()
        >>> # Простой пайплайн (TTS + клонирование через SeedVC)
        >>> result = model.generate(
        ...     text="Привет!",
        ...     reference_audio="voice.wav",
        ...     language="ru",
        ... )
        >>> result.save("output.wav")
        >>>
        >>> # Полный доступ к OmniVoice напрямую
        >>> audio = model.tts.design("Hello!", language="en", female=True)
        >>> audio = model.tts.generate("Hi!", instruct="female", speed=1.2)
    """

    def __init__(self, config: Optional[HybridVoiceConfig] = None):
        """
        Инициализирует HybridVoice.

        Модели НЕ загружаются сразу — они загрузятся лениво
        при первом вызове .generate() (или вручную через .load()).

        Args:
            config: Конфигурация. Если None — используется
                HybridVoiceConfig() со значениями по умолчанию.

        Example:
            >>> model = HybridVoice()
            >>> # или с кастомной конфигурацией:
            >>> config = HybridVoiceConfig(device="cuda", diffusion_steps=100)
            >>> model = HybridVoice(config)
        """
        self.config = config or HybridVoiceConfig()

        self.tts = OmniVoiceTTS(self.config) if self.config.tts_enabled else None
        self.vc = SeedVC(self.config) if self.config.vc_enabled else None

        logger.info("HybridVoice инициализирован.")

    def load(self) -> "HybridVoice":
        """
        Предзагружает все модели в память.

        Вызывать необязательно — модели загрузятся автоматически
        при первом .generate(). Но для избежания задержки на первом
        вызове можно вызвать .load() заранее.

        Returns:
            self (для цепочки вызовов).

        Example:
            >>> model = HybridVoice().load()
        """
        if self.tts is not None:
            self.tts.load()
        if self.vc is not None:
            self.vc.load()
        return self

    def generate(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        female: bool = False,
        duration: Optional[float] = None,
        speed: Optional[float] = None,
        skip_voice_conversion: bool = False,
        # --- Параметры SeedVC (можно менять на лету) ---
        diffusion_steps: Optional[int] = None,
        length_adjust: Optional[float] = None,
        inference_cfg_rate: Optional[float] = None,
        auto_f0_adjust: Optional[bool] = None,
        pitch_shift: Optional[int] = None,
        **gen_kwargs,
    ) -> AudioResult:
        """
        Генерирует речь из текста с клонированием голоса.

        ЛОГИКА ПАЙПЛАЙНА (как в Dublaj):
            Даже если пользователь делает TTS с клонированием, OmniVoice
            ВСЕГДА генерирует речь с голосом ПО УМОЛЧАНИЮ (Voice Design,
            обычный male/female на нужном языке) — БЕЗ референс-аудио.
            Затем SeedVC берёт reference_audio и НАКЛАДЫВАЕТ этот голос
            поверх сгенерированной речи.

        Шаги:
            1. OmniVoice [Voice Design] — текст → речь с default-голосом.
            2. SeedVC [клонирование] — накладывает голос из reference_audio.

        Если reference_audio=None — возвращается чистый результат OmniVoice
        (Voice Design с указанным voice/instruct).

        Args:
            text: Текст для синтеза (обязательно).
            reference_audio: Путь к аудио-образцу голоса для КЛОНИРОВАНИЯ.
                Этот голос SeedVC наложит на речь от OmniVoice.
                Если None — клонирование не выполняется, вернётся только TTS.
            voice: Instruct-голос для TTS ("male", "female", ...).
                Если None — подбирается по языку (male по умолчанию).
                ВНИМАНИЕ: при клонировании этот голос — лишь "заготовка",
                финальный тембр задаёт reference_audio через SeedVC.
            language: Код языка ISO 639 ("ru", "en", "tr"...).
            female: Если True и voice=None — использовать женский голос.
            duration: Фиксированная длительность (сек). None = авто.
            speed: Скорость речи (>1.0 быстрее, <1.0 медленнее).
            skip_voice_conversion: Если True — пропустить SeedVC
                и вернуть только TTS-результат.

            --- Параметры SeedVC (переопределяют config на этот вызов) ---
            diffusion_steps: Число шагов диффузии SeedVC.
                None = использовать значение из config (по умолчанию 100).
            length_adjust: Коэффициент длины/скорости SeedVC.
                None = из config (по умолчанию 1.0).
            inference_cfg_rate: Сходство с референсом (0.0-1.0).
                None = из config (по умолчанию 0.8). Выше = ближе к референсу.
            auto_f0_adjust: Автоподстройка высоты голоса.
                None = из config (по умолчанию True).
            pitch_shift: Сдвиг тона в полутонах.
                None = из config (по умолчанию 0).

            **gen_kwargs: Доп. параметры генерации OmniVoice
                (num_step, guidance_scale, t_shift и т.д.).

        Returns:
            AudioResult с готовым аудио. Используйте .save() для
            сохранения или .audio для доступа к numpy-массиву.

        Raises:
            ValueError: Если не включён ни TTS, ни VC.

        Example:
            >>> # TTS с клонированием голоса
            >>> result = model.generate(
            ...     text="Привет, мир!",
            ...     reference_audio="voice.wav",   # голос для клонирования
            ...     language="ru",
            ... )
            >>> result.save("output.wav")
            >>>
            >>> # С переопределением параметров SeedVC на этот вызов
            >>> result = model.generate(
            ...     text="Привет!",
            ...     reference_audio="voice.wav",
            ...     language="ru",
            ...     diffusion_steps=50,        # быстрее (вместо 100)
            ...     inference_cfg_rate=0.9,    # ближе к референсу
            ...     pitch_shift=2,             # +2 полутона
            ... )
        """
        start_time = time.time()

        if self.tts is None and self.vc is None:
            raise ValueError(
                "В конфигурации отключены и TTS, и Voice Conversion. "
                "Включите хотя бы один из них."
            )

        _language = language if language is not None else self.config.language

        # ============================================================
        # ШАГ 1: TTS (OmniVoice, режим Voice Design)
        # ------------------------------------------------------------
        # ВАЖНО: даже при клонировании OmniVoice генерирует речь с
        # голосом ПО УМОЛЧАНИЮ (design). Референс сюда НЕ передаётся —
        # клонирование делает SeedVC на шаге 2.
        # ============================================================
        if self.tts is None:
            raise ValueError(
                "TTS отключён. Для пайплайна нужен OmniVoice TTS. "
                "Включите tts_enabled=True."
            )

        logger.info("=" * 50)
        logger.info("ШАГ 1/2: Синтез речи (OmniVoice Voice Design, default голос)")
        logger.info("=" * 50)

        tts_audio = self.tts.design(
            text=text,
            voice=voice,
            language=_language,
            female=female,
            duration=duration,
            speed=speed,
            **gen_kwargs,
        )
        current_audio = tts_audio
        current_sr = self.tts.sampling_rate

        # ============================================================
        # ШАГ 2: Voice Conversion (SeedVC F0) — наложение голоса
        # ------------------------------------------------------------
        # SeedVC берёт reference_audio и накладывает его тембр на
        # речь, сгенерированную OmniVoice с default-голосом.
        # ============================================================
        if self.vc is not None and not skip_voice_conversion:
            if reference_audio is None:
                logger.info(
                    "reference_audio не указан — возвращаю чистый TTS (без клонирования)."
                )
            else:
                logger.info("=" * 50)
                logger.info("ШАГ 2/2: Наложение голоса (SeedVC F0)")
                logger.info("=" * 50)

                current_audio = self.vc.convert(
                    source=current_audio,
                    target=reference_audio,
                    source_sr=current_sr,
                    # Параметры SeedVC (None → берётся значение из config)
                    diffusion_steps=diffusion_steps,
                    length_adjust=length_adjust,
                    inference_cfg_rate=inference_cfg_rate,
                    auto_f0_adjust=auto_f0_adjust,
                    pitch_shift=pitch_shift,
                )
                current_sr = self.vc.sr

        # ============================================================
        # Результат
        # ============================================================
        elapsed = time.time() - start_time
        logger.info(f"Генерация завершена за {elapsed:.2f} сек.")

        metadata = {
            "text": text,
            "language": _language,
            "voice": voice,
            "female": female,
            "reference_audio": reference_audio,
            "elapsed_seconds": elapsed,
            "tts_model": self.config.tts_model,
            "vc_checkpoint": self.config.vc_checkpoint,
        }

        return AudioResult(
            audio=current_audio,
            sample_rate=current_sr,
            metadata=metadata,
        )

    def unload(self) -> None:
        """
        Выгружает все модели из памяти и освобождает VRAM.

        Example:
            >>> model.unload()
        """
        if self.tts is not None:
            self.tts.unload()
        if self.vc is not None:
            self.vc.unload()
        logger.info("Все модели выгружены.")

    def __repr__(self) -> str:
        """
        Строковое представление модели.

        Returns:
            Строка с информацией о конфигурации.
        """
        return (
            f"HybridVoice(\n"
            f"  tts={self.config.tts_model if self.tts else 'disabled'},\n"
            f"  vc={self.config.vc_checkpoint if self.vc else 'disabled'},\n"
            f"  device={self.config.device or 'auto'}\n"
            f")"
        )
