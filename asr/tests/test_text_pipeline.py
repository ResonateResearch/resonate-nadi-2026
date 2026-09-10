"""Synthetic text-only release checks. No competition data or model execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import local_score
import ortho_canon
import rover_combine


class TextPipelineTests(unittest.TestCase):
    def test_normalization_keeps_letter_forms(self):
        self.assertEqual(local_score.normalize("مَرْحَبًا، يا أختي؟"), "مرحبا يا أختي")
        self.assertEqual(local_score.normalize("أ إ آ ى ة"), "أ إ آ ى ة")

    def test_pooled_scoring_excludes_empty_references(self):
        wer, cer, n = local_score.score_pairs([("أنا هنا", "أنا هون"), ("", "كلام")])
        self.assertEqual(n, 1)
        self.assertEqual(wer, 0.5)
        self.assertGreater(cer, 0)

    def test_pivot_alignment_drops_insertions(self):
        self.assertEqual(rover_combine.align_to_pivot(["أنا", "هنا"], ["أنا", "الآن", "هنا"]), ["أنا", "هنا"])
        self.assertEqual(rover_combine.align_to_pivot(["أنا", "هنا"], ["أنا"]), ["أنا", ""])

    def test_canonicalizer_uses_explicit_country(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "train.jsonl"
            rows = [{"dialect": "Jordan", "audio": "different/depth/clip.wav", "text": "language Arabic<asr_text>على"}] * 30
            manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            lex = ortho_canon.build_lexicon(manifest, 0.6, 30)
            self.assertEqual(lex["Jordan"]["علي"], "على")
            manifest.write_text(json.dumps({"audio": "any/clip.wav", "text": "على"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ortho_canon.build_lexicon(manifest, 0.6, 30)

    def test_rover_cli_nine_explicit_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            systems = []
            for i in range(9):
                system = root / str(i)
                system.mkdir()
                row = {"utt_id": "synthetic", "dataset": "Jordan", "ref": "أنا هون", "hyp": "أنا هنا" if i == 0 else "أنا هون"}
                (system / "hyp.shard_0").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
                systems.append(str(system))
            out = root / "combined"
            subprocess.run([sys.executable, str(SCRIPTS / "rover_combine.py"), "--systems", *systems,
                            "--weights", "1.30", "1.10", "1.05", "1.00", "1.00", "0.90", "0.85", "0.80", "0.75",
                            "--out", str(out)], check=True, capture_output=True, text=True)
            result = json.loads((out / "hyp.shard_0").read_text(encoding="utf-8"))
            self.assertEqual(result["hyp"], "أنا هون")
            self.assertTrue((out / "wer_report_local.json").is_file())


if __name__ == "__main__":
    unittest.main()
