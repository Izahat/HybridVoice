"""
Конфигурация библиотеки HybridVoice.

Параметры по умолчанию:
    - SeedVC V1 F0 Base (44100 Hz, RMVPE, BigVGAN 44k)
    - OmniVoice в режиме Voice Design (instruct)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from .utils import get_best_device, resolve_device


logger = logging.getLogger(__name__)


@dataclass
class HybridVoiceConfig:
    """
    Главная конфигурация HybridVoice.

    Управляет всем пайплайном: TTS (OmniVoice) → Voice Conversion (SeedVC F0).

    Attributes:
        tts_model: Имя модели TTS на HuggingFace или локальный путь.
        vc_checkpoint: Имя чекпоинта SeedVC на HuggingFace.
        vc_config: Имя конфигурационного файла SeedVC на HuggingFace.
        device: Устройство для инференса ("cuda", "cpu", "mps").
            Если None — определится автоматически.
        dtype: Тип данных для моделей ("float16", "float32").
        cache_dir: Папка для кэширования весов моделей.
        tts_enabled: Включён ли этап TTS.
        vc_enabled: Включён ли этап Voice Conversion.

        --- Параметры TTS (OmniVoice) ---
        num_steps: Количество шагов диффузии для TTS (32 = макс качество).
        default_voice: Голос по умолчанию для Voice Design режима.
        language: Язык по умолчанию (ISO 639: "ru", "en", "tr"...).

        --- Параметры Voice Conversion (SeedVC F0) ---
        diffusion_steps: Шаги диффузии для SeedVC (100 = макс качество).
        inference_cfg_rate: Сходство с референсом (по умолчанию 0.8).
        length_adjust: Коэффициент длины/скорости (1.0 = норма).
        f0_condition: Использовать F0-модель (всегда True).
        auto_f0_adjust: Автоподстройка высоты голоса (всегда True).
        pitch_shift: Сдвиг тона в полутонах (0 = без сдвига).

    Example:
        >>> config = HybridVoiceConfig(device="cuda")
        >>> model = HybridVoice(config)
    """

    # --- Модели ---
    tts_model: str = "k2-fsa/OmniVoice"
    vc_repo: str = "Plachta/Seed-VC"
    vc_checkpoint: str = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"
    vc_config: str = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"

    # --- Устройство и точность ---
    device: Optional[str] = None
    dtype: str = "float16"
    cache_dir: Optional[str] = None

    # --- Флаги пайплайна ---
    tts_enabled: bool = True
    vc_enabled: bool = True

    # --- Параметры TTS (OmniVoice) ---
    num_steps: int = 32
    default_voice: str = "male"
    language: Optional[str] = None

    # --- Параметры Voice Conversion (SeedVC F0) ---
    diffusion_steps: int = 100
    inference_cfg_rate: float = 0.8
    length_adjust: float = 1.0
    f0_condition: bool = True
    auto_f0_adjust: bool = True
    pitch_shift: int = 0

    # --- Валидация reference-аудио ---
    min_reference_duration: float = 0.5
    max_reference_duration: float = 25.0
    silence_threshold: float = 1e-5

    def __post_init__(self) -> None:
        """Проверяет конфигурацию до создания и загрузки моделей."""
        if not isinstance(self.num_steps, int) or isinstance(self.num_steps, bool):
            raise TypeError("num_steps должен быть целым числом.")
        if self.num_steps <= 0:
            raise ValueError("num_steps должен быть больше 0.")

        self.validate_vc_parameters()

        if not isinstance(self.min_reference_duration, (int, float)) or isinstance(
            self.min_reference_duration, bool
        ):
            raise TypeError("min_reference_duration должен быть числом.")
        if not math.isfinite(self.min_reference_duration) or self.min_reference_duration <= 0:
            raise ValueError("min_reference_duration должен быть больше 0.")
        if not isinstance(self.max_reference_duration, (int, float)) or isinstance(
            self.max_reference_duration, bool
        ):
            raise TypeError("max_reference_duration должен быть числом.")
        if not math.isfinite(self.max_reference_duration):
            raise ValueError("max_reference_duration должен быть конечным числом.")
        if self.max_reference_duration < self.min_reference_duration:
            raise ValueError(
                "max_reference_duration должен быть не меньше min_reference_duration."
            )
        if not isinstance(self.silence_threshold, (int, float)) or isinstance(
            self.silence_threshold, bool
        ):
            raise TypeError("silence_threshold должен быть числом.")
        if not math.isfinite(self.silence_threshold) or self.silence_threshold < 0:
            raise ValueError("silence_threshold не может быть отрицательным.")

        if not isinstance(self.dtype, str):
            raise TypeError("dtype должен быть строкой.")
        if self.dtype.lower().strip() not in {"float16", "fp16", "float32", "fp32"}:
            raise ValueError(
                f"Неизвестный dtype: '{self.dtype}'. "
                "Допустимые: float16, fp16, float32, fp32."
            )

        if not isinstance(self.f0_condition, bool):
            raise TypeError("f0_condition должен быть bool.")
        if not self.f0_condition:
            raise ValueError(
                "Текущая реализация HybridVoice поддерживает только "
                "SeedVC F0; f0_condition должен быть True."
            )
        if not isinstance(self.auto_f0_adjust, bool):
            raise TypeError("auto_f0_adjust должен быть bool.")
        if not isinstance(self.pitch_shift, int) or isinstance(self.pitch_shift, bool):
            raise TypeError("pitch_shift должен быть целым числом.")

        # Для явно указанного устройства проверяем и синтаксис, и доступность.
        if self.device is not None:
            resolve_device(self.device)

    def validate_vc_parameters(
        self,
        diffusion_steps: Optional[int] = None,
        length_adjust: Optional[float] = None,
        inference_cfg_rate: Optional[float] = None,
    ) -> tuple[int, float, float]:
        """Возвращает проверенные фактические параметры SeedVC."""
        steps = self.diffusion_steps if diffusion_steps is None else diffusion_steps
        length = self.length_adjust if length_adjust is None else length_adjust
        cfg_rate = (
            self.inference_cfg_rate
            if inference_cfg_rate is None
            else inference_cfg_rate
        )

        if not isinstance(steps, int) or isinstance(steps, bool):
            raise TypeError("diffusion_steps должен быть целым числом.")
        if steps <= 0:
            raise ValueError("diffusion_steps должен быть больше 0.")
        if not isinstance(length, (int, float)) or isinstance(length, bool):
            raise TypeError("length_adjust должен быть числом.")
        if not math.isfinite(length) or length <= 0:
            raise ValueError("length_adjust должен быть больше 0.")
        if not isinstance(cfg_rate, (int, float)) or isinstance(cfg_rate, bool):
            raise TypeError("inference_cfg_rate должен быть числом.")
        if not math.isfinite(cfg_rate) or not 0.0 <= cfg_rate <= 1.0:
            raise ValueError("inference_cfg_rate должен находиться в диапазоне [0, 1].")

        return steps, float(length), float(cfg_rate)

    # --- Sample rates (фиксированные для моделей) ---
    @property
    def tts_sample_rate(self) -> int:
        """
        Частота дискретизации OmniVoice.

        Returns:
            24000 Hz — фиксированный sample rate OmniVoice.
        """
        return 24000

    @property
    def vc_sample_rate(self) -> int:
        """
        Частота дискретизации SeedVC F0 модели.

        Returns:
            44100 Hz — фиксированный sample rate F0-модели.
        """
        return 44100

    def get_torch_dtype(self, device: Optional[str] = None):
        """
        Возвращает torch.dtype на основе строкового поля self.dtype.

        Returns:
            torch.float16 или torch.float32.

        Raises:
            ValueError: Если строка dtype не распознана.
        """
        import torch

        dtype_name = self.dtype.lower().strip()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved_device = device or self.device or get_best_device()
        if str(resolved_device).split(":", 1)[0] == "cpu" and dtype_name in {
            "float16",
            "fp16",
        }:
            logger.info("CPU не использует float16: автоматически выбран float32.")
            return torch.float32

        return dtype_map[dtype_name]
