"""
Конфигурация библиотеки HybridVoice.

Параметры по умолчанию:
    - SeedVC V1 F0 Base (44100 Hz, RMVPE, BigVGAN 44k)
    - OmniVoice в режиме Voice Design (instruct)
"""

from dataclasses import dataclass, field
from typing import Optional


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

    def get_torch_dtype(self):
        """
        Возвращает torch.dtype на основе строкового поля self.dtype.

        Returns:
            torch.float16 или torch.float32.

        Raises:
            ValueError: Если строка dtype не распознана.
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(
                f"Неизвестный dtype: '{self.dtype}'. "
                f"Допустимые: {list(dtype_map.keys())}"
            )
        return dtype_map[self.dtype]
