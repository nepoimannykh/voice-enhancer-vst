import unittest
from pathlib import Path

from voice_enh.cli import audio_args, base_filters, loudnorm_filter, matched_loudnorm
from voice_enh.clearvoice_runner import project_dir


class CliTests(unittest.TestCase):
    def test_codec_by_extension(self):
        self.assertEqual(audio_args(Path("voice.wav")), ["-c:a", "pcm_s24le"])
        self.assertIn("libmp3lame", audio_args(Path("voice.mp3")))

    def test_second_pass_loudness_parameters(self):
        measured = {"input_i": "-24", "input_lra": "4", "input_tp": "-3", "input_thresh": "-34", "target_offset": "0.2"}
        result = loudnorm_filter("studio", measured)
        self.assertIn("measured_I=-24", result)
        self.assertIn("linear=true", result)

    def test_source_loudness_is_preserved(self):
        self.assertTrue(matched_loudnorm(-21.3).startswith("loudnorm=I=-21.3:"))

    def test_clearvoice_checkpoint_is_anchored_to_project(self):
        checkpoint = project_dir() / "checkpoints" / "MossFormer2_SE_48K" / "last_best_checkpoint.pt"
        self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
