import json
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.eval_denoiser import _json_default


class TestEvalDenoiserJsonExport(unittest.TestCase):
    def test_json_default_handles_numpy_scalars(self) -> None:
        payload = {
            "i64": np.int64(7),
            "f64": np.float64(0.25),
            "b": np.bool_(True),
            "arr": np.array([np.int64(1), np.int64(2)]),
        }

        # Without a default serializer, stdlib json cannot handle numpy types.
        with self.assertRaises(TypeError):
            json.dumps(payload, allow_nan=False)

        out = json.dumps(payload, allow_nan=False, default=_json_default)
        decoded = json.loads(out)
        self.assertEqual(decoded["i64"], 7)
        self.assertEqual(decoded["f64"], 0.25)
        self.assertEqual(decoded["b"], True)
        self.assertEqual(decoded["arr"], [1, 2])

    def test_json_default_handles_pandas_timestamp_and_na(self) -> None:
        ts = pd.Timestamp("2025-12-24T12:34:56Z")
        payload = {"ts": ts, "na": pd.NA, "dt": datetime(2025, 12, 24, tzinfo=timezone.utc)}

        out = json.dumps(payload, allow_nan=False, default=_json_default)
        decoded = json.loads(out)
        self.assertEqual(decoded["na"], None)
        self.assertTrue(decoded["ts"].startswith("2025-12-24T12:34:56"))
        self.assertTrue(decoded["dt"].startswith("2025-12-24T00:00:00") or decoded["dt"].startswith("2025-12-24T"))


if __name__ == "__main__":
    unittest.main()

