"""Быстрые unit-тесты валидации без загрузки весов моделей."""

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hybridvoice import HybridVoice, HybridVoiceConfig
from hybridvoice.utils import validate_reference_audio
from hybridvoice.vc import SeedVC


class ConfigValidationTests(unittest.TestCase):
    def test_cpu_automatically_uses_float32(self):
        config = HybridVoiceConfig(device="cpu", dtype="float16")
        self.assertEqual(config.get_torch_dtype("cpu"), torch.float32)

    def test_invalid_vc_parameters_fail_early(self):
        invalid_configs = (
            {"diffusion_steps": 0},
            {"length_adjust": 0},
            {"length_adjust": float("nan")},
            {"inference_cfg_rate": -0.1},
            {"inference_cfg_rate": 1.1},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                HybridVoiceConfig(device="cpu", **kwargs)

    def test_only_f0_seedvc_is_supported(self):
        with self.assertRaises(ValueError):
            HybridVoiceConfig(device="cpu", f0_condition=False)

    def test_empty_text_fails_without_loading_tts(self):
        model = HybridVoice(
            HybridVoiceConfig(device="cpu", dtype="float32", vc_enabled=False)
        )
        with self.assertRaises(ValueError):
            model.generate("   ")
        self.assertFalse(model.tts.is_loaded)

    def test_missing_reference_fails_without_loading_models(self):
        model = HybridVoice(HybridVoiceConfig(device="cpu", dtype="float32"))
        with self.assertRaises(FileNotFoundError):
            model.generate("Hello", reference_audio="definitely-missing.wav")
        self.assertFalse(model.tts.is_loaded)
        self.assertFalse(model.vc.is_loaded)


class ReferenceAudioValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.sample_rate = 16000

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, name: str, waveform: np.ndarray) -> Path:
        path = self.root / name
        sf.write(path, waveform, self.sample_rate)
        return path

    def test_valid_reference_is_accepted(self):
        time = np.arange(self.sample_rate, dtype=np.float32) / self.sample_rate
        path = self._write("voice.wav", 0.1 * np.sin(2 * np.pi * 220 * time))
        self.assertEqual(validate_reference_audio(path), path)

    def test_silent_reference_is_rejected(self):
        path = self._write("silence.wav", np.zeros(self.sample_rate, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "не содержит слышимого сигнала"):
            validate_reference_audio(path)

    def test_short_reference_is_rejected(self):
        path = self._write("short.wav", np.ones(1000, dtype=np.float32) * 0.1)
        with self.assertRaisesRegex(ValueError, "слишком короткое"):
            validate_reference_audio(path)


class F0SafetyTests(unittest.TestCase):
    def test_empty_reference_f0_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "reference-аудио"):
            SeedVC._prepare_shifted_f0(
                f0_reference=torch.zeros(10),
                f0_source=torch.full((10,), 120.0),
                auto_f0_adjust=True,
                pitch_shift=0,
            )

    def test_empty_source_f0_returns_finite_fallback(self):
        source = torch.zeros(10)
        result = SeedVC._prepare_shifted_f0(
            f0_reference=torch.full((10,), 180.0),
            f0_source=source,
            auto_f0_adjust=True,
            pitch_shift=0,
        )
        self.assertTrue(torch.equal(result, source))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
