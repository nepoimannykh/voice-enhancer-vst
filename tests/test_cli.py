import unittest
from pathlib import Path

import numpy as np

from voice_enh.cli import audio_args, base_filters, describe_dereverb, matched_loudnorm, parser
from voice_enh.buttercomp_runner import ButterComp2
from voice_enh.studio_dsp_runner import DeBess, highpass_80hz, peaking_sos, transparent_eq
from voice_enh.clearvoice_runner import project_dir


class CliTests(unittest.TestCase):
    def test_codec_by_extension(self):
        self.assertEqual(audio_args(Path("voice.wav")), ["-c:a", "pcm_s24le"])
        self.assertIn("libmp3lame", audio_args(Path("voice.mp3")))

    def test_post_processing_uses_u87_profile_without_gate_or_click_repair(self):
        filters = base_filters()
        self.assertEqual(filters[0], "highpass=f=80:p=2")
        self.assertTrue(any(value.startswith("equalizer=f=900:") for value in filters))
        self.assertFalse(any(value.startswith("aexciter=") for value in filters))
        self.assertFalse(any(value.startswith("acompressor=") for value in filters))
        deessers = [value for value in filters if value.startswith("deesser=")]
        self.assertEqual(deessers, ["deesser=i=0.45:m=0.30:f=0.4200"])
        self.assertFalse(any(value.startswith(("agate=", "adeclick=")) for value in filters))
        eq = next(i for i, value in enumerate(filters) if value.startswith("equalizer="))
        deesser = next(i for i, value in enumerate(filters) if value.startswith("deesser="))
        self.assertLess(eq, deesser)

    def test_deesser_can_be_disabled(self):
        self.assertFalse(any(value.startswith(("deesser=", "adynamicequalizer=")) for value in base_filters(False)))

    def test_dereverb_attenuation_limit_is_configurable(self):
        args = parser().parse_args(["voice.wav", "--dereverb-attn-limit-db", "8"])
        self.assertEqual(args.dereverb_attn_limit_db, 8.0)

    def test_dereverb_level_is_human_readable(self):
        self.assertEqual(describe_dereverb(3), "light: 3 dB limit, 29% enhanced blend")
        self.assertIn("100% enhanced", describe_dereverb(None))

    def test_source_loudness_is_preserved(self):
        self.assertTrue(matched_loudnorm(-21.3).startswith("loudnorm=I=-21.3:"))

    def test_buttercomp_is_finite_and_reports_reduction(self):
        time = np.arange(4800) / 48000
        audio = 0.35 * np.sin(2 * np.pi * 220 * time)
        processed, stats = ButterComp2(48000).process(audio)
        self.assertEqual(processed.shape, audio.shape)
        self.assertTrue(np.isfinite(processed).all())
        self.assertGreater(stats["p95_reduction_db"], 0)

    def test_studio_eq_and_debess_are_finite(self):
        time = np.arange(4800) / 48000
        audio = 0.2 * np.sin(2 * np.pi * 220 * time) + 0.03 * np.sin(2 * np.pi * 6000 * time)
        bands = [{"frequency": 900, "q": 0.7, "gain_db": 1.5}]
        highpassed = highpass_80hz(audio, 48000)
        equalized = transparent_eq(highpassed, 48000, bands)
        processed, stats = DeBess(48000, 6000).process(equalized)
        self.assertEqual(peaking_sos(48000, 900, 0.7, 1.5).shape, (1, 6))
        self.assertTrue(np.isfinite(processed).all())
        self.assertTrue(np.isfinite(highpassed).all())
        self.assertGreaterEqual(stats["max_reduction_db"], 0)

    def test_clearvoice_checkpoint_is_anchored_to_project(self):
        checkpoint = project_dir() / "checkpoints" / "MossFormer2_SE_48K" / "last_best_checkpoint.pt"
        self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
