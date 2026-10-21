import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qr-code-generator" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import batch_generate


class BatchGenerateTests(unittest.TestCase):
    def test_generates_each_valid_row_in_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "codes.csv"
            output_dir = Path(temp_dir) / "output"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "url", "label"])
                writer.writeheader()
                writer.writerows([
                    {"id": "first", "url": "https://example.com/1", "label": "First"},
                    {"id": "", "url": "https://example.com/skip", "label": ""},
                    {"id": "second", "url": "https://example.com/2", "label": ""},
                ])

            with patch.object(
                sys,
                "argv",
                [
                    "batch_generate.py",
                    "--csv",
                    str(csv_path),
                    "--outdir",
                    str(output_dir),
                    "--format",
                    "both",
                    "--error",
                    "Q",
                    "--show-url",
                ],
            ), patch.object(batch_generate, "generate_qr", side_effect=lambda url, **_: url) as generate:
                batch_generate.main()

            self.assertEqual(generate.call_count, 2)
            generate.assert_any_call(
                "https://example.com/1",
                png=str(output_dir / "first.png"),
                svg=str(output_dir / "first.svg"),
                error="Q",
                caption_label="First",
                show_url=True,
            )
            generate.assert_any_call(
                "https://example.com/2",
                png=str(output_dir / "second.png"),
                svg=str(output_dir / "second.svg"),
                error="Q",
                caption_label=None,
                show_url=True,
            )


if __name__ == "__main__":
    unittest.main()
