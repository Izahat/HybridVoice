"""
Модуль Voice Conversion на базе SeedVC F0 Base.

Использует следующую конфигурацию моделей:
    - Модель: Seed-VC V1 F0 Base (seed-uvit-whisper-base, 200M параметров)
    - Sample rate: 44100 Hz
    - F0 conditioning: ВСЕГДА ВКЛЮЧЕН
    - Чекпоинт: DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth
    - Vocoder: BigVGAN (nvidia/bigvgan_v2_44khz_128band_512x)
    - Pitch extractor: RMVPE

Параметры по умолчанию:
    diffusion_steps=100, length_adjust=1.0, inference_cfg_rate=0.8,
    auto_f0_adjust=True, f0_condition=True, pitch_shift=0
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import HybridVoiceConfig
from .utils import get_best_device

logger = logging.getLogger(__name__)


class SeedVC:
    """
    Обёртка для модели Voice Conversion SeedVC F0 Base.

    Загружает DiT-модель (F0), Whisper, CAMPPlus, RMVPE и BigVGAN,
    затем преобразует голос из source в тембр target с сохранением
    высоты тона (F0 conditioning).

    Attributes:
        config: Конфигурация HybridVoice.
        device: Устройство для инференса.
        sr: Частота дискретизации (44100 для F0-модели).
        is_loaded: Загружена ли модель.

    Example:
        >>> vc = SeedVC(config)
        >>> result = vc.convert(source=tts_audio, target="voice.wav", source_sr=24000)
    """

    def __init__(self, config: HybridVoiceConfig):
        """
        Инициализирует обёртку SeedVC (модель ещё НЕ загружается).

        Args:
            config: Конфигурация HybridVoice.
        """
        self.config = config
        self.device = config.device or get_best_device()
        self.is_loaded = False

        # Кэш весов: СТАНДАРТНЫЙ HF-кэш (как у transformers/diffusers).
        # config.cache_dir позволяет переопределить папку при желании.
        self.custom_cache_dir = config.cache_dir  # None = стандартный HF-кэш

        # Атрибуты моделей (заполняются в load())
        self.model = None
        self.semantic_fn = None
        self.f0_fn = None
        self.vocoder_fn = None
        self.campplus_model = None
        self.to_mel = None
        self.sr = config.vc_sample_rate  # 44100 для F0-модели
        self.hop_length = 512

    @property
    def checkpoint_dir(self) -> Path:
        """
        Фактическая папка кэша весов.

        Возвращает:
            Путь к папке кэша: config.cache_dir, если задан,
            иначе стандартный HF-кэш (например,
            ~/.cache/huggingface/hub на Linux или
            C:\\Users\\<user>\\.cache\\huggingface\\hub на Windows).
        """
        if self.custom_cache_dir:
            return Path(self.custom_cache_dir)
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)

    def _load_from_hf(
        self,
        repo_id: str,
        model_filename: str,
        config_filename: Optional[str] = None,
    ):
        """
        Скачивает файл(ы) модели с HuggingFace в стандартный HF-кэш.

        Это наша замена hf_utils.load_custom_model_from_hf из seed-vc,
        которая зависела от рабочей директории ("./checkpoints").
        Все веса SeedVC лежат в одном кэше с OmniVoice/Whisper/BigVGAN.

        При повторном запуске файлы берутся из кэша, повторного
        скачивания не происходит.

        Args:
            repo_id: Имя репозитория на HuggingFace
                (например, "Plachta/Seed-VC").
            model_filename: Имя файла весов
                (например, "DiT_seed_v2_..._ft_ema.pth").
            config_filename: Имя конфиг-файла (опционально).

        Returns:
            Путь к весам, или кортеж (путь к весам, путь к конфигy),
            если config_filename указан.
        """
        from huggingface_hub import hf_hub_download

        # cache_dir=None → hf_hub_download использует стандартный HF-кэш
        cache_dir = self.custom_cache_dir  # None если не задан в config

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            cache_dir=cache_dir,
        )
        if config_filename is None:
            return model_path

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            cache_dir=cache_dir,
        )
        return model_path, config_path

    def load(self) -> "SeedVC":
        """
        Загружает все компоненты SeedVC F0 в память.

        Загружает:
            - DiT F0 модель (основная модель конверсии, 44100 Hz)
            - Whisper (семантические признаки)
            - CAMPPlus (эмбеддинг говорящего)
            - RMVPE (экстрактор высоты тона F0)
            - BigVGAN 44k (вокодер)

        Returns:
            self (для цепочки вызовов).

        Example:
            >>> vc = SeedVC(config).load()
        """
        if self.is_loaded:
            logger.info("SeedVC уже загружена, пропускаем.")
            return self

        import yaml

        logger.info("Загрузка SeedVC F0 Base...")
        logger.info(f"Кэш весов SeedVC: {self.checkpoint_dir}")

        from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch

        # ============================================================
        # 1. DiT F0 модель (44100 Hz)
        # ============================================================
        logger.info(f"Загрузка DiT F0: {self.config.vc_checkpoint}")
        dit_checkpoint_path, dit_config_path = self._load_from_hf(
            self.config.vc_repo,
            self.config.vc_checkpoint,
            self.config.vc_config,
        )

        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        self.model = build_model(model_params, stage="DiT")
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]

        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # --- Mel-спектrogramма ---
        from seed_vc.modules.audio import mel_spectrogram

        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": self.hop_length,
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000,
            "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        # ============================================================
        # 2. Whisper (семантические признаки)
        # ============================================================
        logger.info("Загрузка Whisper...")
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(
            whisper_name, torch_dtype=torch.float16
        ).to(self.device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            """Извлекает семантические признаки через Whisper encoder.

            Устойчив к разным версиям transformers: если приватный метод
            _mask_input_features отсутствует (новые версии), используем
            прямую передачу attention_mask в encoder.
            """
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = ori_inputs.input_features
            attention_mask = ori_inputs.attention_mask

            # В старых transformers маскизация делается отдельным методом
            if hasattr(whisper_model, "_mask_input_features"):
                input_features = whisper_model._mask_input_features(
                    input_features, attention_mask=attention_mask
                )
                encoder_extra = {}
            else:
                # В новых transformers передаём attention_mask в encoder
                encoder_extra = {"attention_mask": attention_mask}

            input_features = input_features.to(self.device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **encoder_extra,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori

        self.semantic_fn = semantic_fn

        # ============================================================
        # 3. RMVPE (экстрактор F0)
        # ============================================================
        logger.info("Загрузка RMVPE (F0 extractor)...")
        from seed_vc.modules.rmvpe import RMVPE

        rmvpe_path = self._load_from_hf(
            "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
        )
        f0_extractor = RMVPE(rmvpe_path, is_half=False, device=self.device)
        self.f0_fn = f0_extractor.infer_from_audio

        # ============================================================
        # 4. CAMPPlus (эмбеддинг говорящего)
        # ============================================================
        logger.info("Загрузка CAMPPlus...")
        from seed_vc.modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt_path = self._load_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        self.campplus_model.eval()
        self.campplus_model.to(self.device)

        # ============================================================
        # 5. BigVGAN 44k (вокодер)
        # ============================================================
        logger.info("Загрузка BigVGAN 44k...")
        from seed_vc.modules.bigvgan import bigvgan

        vocoder_type = model_params.vocoder.type
        if vocoder_type == "bigvgan":
            bigvgan_name = model_params.vocoder.name
            self.vocoder_fn = self._load_bigvgan(bigvgan, bigvgan_name)
            self.vocoder_fn.remove_weight_norm()
            self.vocoder_fn = self.vocoder_fn.eval().to(self.device)
        else:
            raise ValueError(f"Неподдерживаемый тип вокодера: {vocoder_type}")

        self.is_loaded = True
        logger.info(
            f"SeedVC F0 загружена. SR={self.sr}, hop={self.hop_length}"
        )
        return self

    def _load_bigvgan(self, bigvgan_module, bigvgan_name: str):
        """
        Загружает BigVGAN в обход BigVGAN.from_pretrained().

        В seed-vc метод _from_pretrained() BigVGAN использует аргументы
        proxies/resume_download, которые были убраны в новых версиях
        huggingface_hub (>=1.0). Этот метод загружает веса вручную через
        hf_hub_download, что работает с любой версией huggingface_hub.

        Args:
            bigvgan_module: Модуль modules.bigvgan.bigvgan (с классом BigVGAN).
            bigvgan_name: Имя модели на HuggingFace
                (например, "nvidia/bigvgan_v2_44khz_128band_512x").

        Returns:
            Загруженный экземпляр BigVGAN (без weight_norm).
        """
        import json

        from huggingface_hub import hf_hub_download

        BigVGAN = bigvgan_module.BigVGAN

        # 1. config.json
        config_file = hf_hub_download(
            repo_id=bigvgan_name, filename="config.json"
        )
        with open(config_file, "r") as f:
            h = bigvgan_module.AttrDict(json.load(f))

        # 2. Инстанцируем модель
        model = BigVGAN(h, use_cuda_kernel=False)

        # 3. Веса
        weights_file = hf_hub_download(
            repo_id=bigvgan_name, filename="bigvgan_generator.pt"
        )
        checkpoint = torch.load(weights_file, map_location="cpu")

        try:
            model.load_state_dict(checkpoint["generator"])
        except RuntimeError:
            # Чекпоинт без weight norm — снимаем и грузим
            model.remove_weight_norm()
            model.load_state_dict(checkpoint["generator"])

        logger.info(f"BigVGAN загружена из {bigvgan_name}")
        return model

    @staticmethod
    def _adjust_f0_semitones(f0_sequence, n_semitones):
        """
        Сдвигает F0-последовательность на заданное число полутонов.

        Args:
            f0_sequence: Тензор или массив со значениями F0.
            n_semitones: Число полутонов для сдвига.

        Returns:
            Сдвинутая F0-последовательность.
        """
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor

    @staticmethod
    def _crossfade(chunk1, chunk2, overlap):
        """
        Применяет плавный кроссфейд между двумя чанками аудио.

        Args:
            chunk1: Предыдущий чанк (numpy).
            chunk2: Текущий чанк (numpy).
            overlap: Длина перекрытия в сэмплах.

        Returns:
            chunk2 с применённым кроссфейдом.
        """
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
            )
        return chunk2

    @torch.no_grad()
    def convert(
        self,
        source: np.ndarray,
        target: str,
        source_sr: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
        length_adjust: Optional[float] = None,
        inference_cfg_rate: Optional[float] = None,
        auto_f0_adjust: Optional[bool] = None,
        pitch_shift: Optional[int] = None,
    ) -> np.ndarray:
        """
        Преобразует голос: контент из source → тембр из target.

        Использует F0 conditioning для сохранения высоты тона.
        Поддерживает длинные аудио через чанки с кроссфейдом.

        Args:
            source: Waveform для конвертации (numpy float32).
                Обычно это выход TTS (24000 Hz).
            target: Путь к референсному аудио (голос-цель).
            source_sr: Sample rate для source. Если None — self.sr.
            diffusion_steps: Шаги диффузии (по умолчанию 100 из config).
            length_adjust: Коэффициент длины (по умолчанию 1.0).
            inference_cfg_rate: Сходство с референсом (по умолчанию 0.8).
            auto_f0_adjust: Автоподстройка F0 (по умолчанию True).
            pitch_shift: Сдвиг тона в полутонах (по умолчанию 0).

        Returns:
            Numpy-массив с конвертированным аудио (sample_rate = self.sr = 44100).

        Example:
            >>> result = vc.convert(tts_audio, target="voice.wav", source_sr=24000)
        """
        if not self.is_loaded:
            self.load()

        import torchaudio
        import librosa

        _steps = diffusion_steps if diffusion_steps is not None else self.config.diffusion_steps
        _length_adjust = length_adjust if length_adjust is not None else self.config.length_adjust
        _cfg_rate = inference_cfg_rate if inference_cfg_rate is not None else self.config.inference_cfg_rate
        _auto_f0 = auto_f0_adjust if auto_f0_adjust is not None else self.config.auto_f0_adjust
        _pitch_shift = pitch_shift if pitch_shift is not None else self.config.pitch_shift

        # --- Загружаем target аудио ---
        ref_audio, _ = librosa.load(target, sr=self.sr, mono=True)
        ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(self.device)
        ref_audio = ref_audio[:, : self.sr * 25]  # макс 25 сек

        # --- Подготавливаем source ---
        if source_sr is not None and source_sr != self.sr:
            source = librosa.resample(
                source, orig_sr=source_sr, target_sr=self.sr
            )
        source_audio = torch.tensor(source).unsqueeze(0).float().to(self.device)

        # --- Параметры чанкования ---
        max_context_window = self.sr // self.hop_length * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * self.hop_length

        # --- Ресемплируем в 16kHz для Whisper и RMVPE ---
        converted_waves_16k = torchaudio.functional.resample(
            source_audio, self.sr, 16000
        )
        ori_waves_16k = torchaudio.functional.resample(
            ref_audio, self.sr, 16000
        )

        # --- Семантические признаки (Whisper) ---
        logger.info("Извлечение семантических признаков (Whisper)...")
        S_alt = self._semantic_with_chunking(converted_waves_16k)
        S_ori = self.semantic_fn(ori_waves_16k)

        # --- Mel-спектrogramмы ---
        mel = self.to_mel(source_audio)
        mel2 = self.to_mel(ref_audio)

        target_lengths = torch.LongTensor(
            [int(mel.size(2) * _length_adjust)]
        ).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # --- Style embedding через CAMPPlus ---
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        # --- F0 экстракция и подстройка ---
        logger.info("Экстракция F0 (RMVPE)...")
        F0_ori = self.f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = self.f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        shifted_log_f0_alt = log_f0_alt.clone()
        if _auto_f0:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if _pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = self._adjust_f0_semitones(
                shifted_f0_alt[F0_alt > 1], _pitch_shift
            )

        # --- Length regulation ---
        cond, _, _, _, _ = self.model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, _, _, _ = self.model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )

        # --- Voice Conversion по чанкам ---
        logger.info(f"Voice Conversion ({_steps} шагов диффузии)...")
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            with torch.autocast(
                device_type=self.device, dtype=torch.float16
            ):
                vc_target = self.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2,
                    style2,
                    None,
                    _steps,
                    inference_cfg_rate=_cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1) :]

            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()
            vc_wave = vc_wave[None, :]

            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                generated_wave_chunks.append(
                    vc_wave[0, :-overlap_wave_len].cpu().numpy()
                )
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
            else:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len

        result = np.concatenate(generated_wave_chunks)
        logger.info(f"Voice Conversion завершена. Длина: {len(result)} сэмплов.")
        return result

    def _semantic_with_chunking(self, waves_16k: torch.Tensor) -> torch.Tensor:
        """
        Извлекает семантические признаки с чанкованием для длинных аудио.

        Whisper обрабатывает максимум 30 секунд за раз. Если аудио длиннее,
        оно разбивается на чанки с перекрытием 5 секунд.

        Args:
            waves_16k: Тензор аудио 16kHz, форма (1, T).

        Returns:
            Тензор семантических признаков, форма (1, T_frames, hidden).
        """
        if waves_16k.size(-1) <= 16000 * 30:
            return self.semantic_fn(waves_16k)

        overlapping_time = 5  # секунд
        S_list = []
        buffer = None
        traversed_time = 0

        while traversed_time < waves_16k.size(-1):
            if buffer is None:
                chunk = waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        waves_16k[
                            :, traversed_time : traversed_time + 16000 * (30 - overlapping_time)
                        ],
                    ],
                    dim=-1,
                )
            S_chunk = self.semantic_fn(chunk)
            if traversed_time == 0:
                S_list.append(S_chunk)
            else:
                S_list.append(S_chunk[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            )

        return torch.cat(S_list, dim=1)

    def unload(self) -> None:
        """
        Выгружает все модели SeedVC из памяти и освобождает VRAM.

        Example:
            >>> vc.unload()
        """
        if not self.is_loaded:
            return

        del self.model
        del self.campplus_model
        del self.vocoder_fn

        self.model = None
        self.semantic_fn = None
        self.f0_fn = None
        self.campplus_model = None
        self.vocoder_fn = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SeedVC выгружена из памяти.")
