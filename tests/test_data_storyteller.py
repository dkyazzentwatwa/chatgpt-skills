import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "data-storyteller" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from data_storyteller import DataStoryteller


class DataStorytellerProfileTests(unittest.TestCase):
    def assert_duplicate_profile(self, frame, expected_count, expected_pct):
        storyteller = DataStoryteller(frame)
        with patch.object(
            storyteller.df,
            "duplicated",
            wraps=storyteller.df.duplicated,
        ) as duplicated:
            profile = storyteller._profile_data()

        self.assertEqual(duplicated.call_count, 1)
        self.assertEqual(profile["duplicates"], expected_count)
        if math.isnan(expected_pct):
            self.assertTrue(math.isnan(profile["duplicate_pct"]))
        else:
            self.assertAlmostEqual(profile["duplicate_pct"], expected_pct)

    def test_empty_dataframe(self):
        self.assert_duplicate_profile(pd.DataFrame(columns=["value"]), 0, math.nan)

    def test_duplicate_free_dataframe(self):
        self.assert_duplicate_profile(pd.DataFrame({"value": [1, 2, 3]}), 0, 0)

    def test_dataframe_with_duplicates(self):
        self.assert_duplicate_profile(pd.DataFrame({"value": [1, 2, 1]}), 1, 100 / 3)


if __name__ == "__main__":
    unittest.main()
