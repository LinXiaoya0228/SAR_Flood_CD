"""Focused tests for E1.3 real smoke runner preflight and boundary guards."""

import unittest
from pathlib import Path
import tempfile
import shutil

from tools.run_e1_3_real_smoke import E1RealSmokeRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/changeguard/e1_3_smoke_event520_v1.json"


class TestE13SmokePreflight(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_e1_3_preflight_"))

    def tearDown(self) -> None:
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_preflight_passes_on_valid_setup(self) -> None:
        runner = E1RealSmokeRunner(
            config_path=CONFIG_PATH,
            output_dir=self.test_dir / "out",
            log_dir=self.test_dir / "log",
            device="cpu",  # CPU for test fixture
        )
        res = runner.validate_preflight(fail_closed=True)
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["target_event"], 520)

    def test_preflight_fails_on_sealed_test_event(self) -> None:
        runner = E1RealSmokeRunner(
            config_path=CONFIG_PATH,
            output_dir=self.test_dir / "out",
            device="cpu",
        )
        runner.target_event = 321  # Forbidden sealed test event
        with self.assertRaises(RuntimeError) as ctx:
            runner.validate_preflight(fail_closed=True)
        self.assertIn("CRITICAL BOUNDARY VIOLATION", str(ctx.exception))

    def test_preflight_fails_on_dirty_output_dir_without_overwrite(self) -> None:
        out = self.test_dir / "dirty_out"
        out.mkdir(parents=True)
        (out / "some_file.txt").write_text("existing")

        runner = E1RealSmokeRunner(
            config_path=CONFIG_PATH,
            output_dir=out,
            device="cpu",
            allow_overwrite=False,
        )
        with self.assertRaises(RuntimeError) as ctx:
            runner.validate_preflight(fail_closed=True)
        self.assertIn("already exists and contains files", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
