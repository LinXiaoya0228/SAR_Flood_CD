"""Unit test wrapping the reproducibility smoke verification suite."""

import unittest
from tools.smoke_reproducibility import (
    check_dependencies,
    check_configs,
    check_synthetic_models,
    check_metric_logic,
    check_canonical_artifacts,
)


class TestReproducibilitySmoke(unittest.TestCase):
    def test_01_dependencies(self):
        self.assertTrue(check_dependencies())

    def test_02_configs(self):
        self.assertTrue(check_configs())

    def test_03_synthetic_models(self):
        self.assertTrue(check_synthetic_models())

    def test_04_metric_logic(self):
        self.assertTrue(check_metric_logic())

    def test_05_canonical_artifacts(self):
        self.assertTrue(check_canonical_artifacts())


if __name__ == "__main__":
    unittest.main()

