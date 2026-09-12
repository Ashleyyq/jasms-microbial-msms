"""Model-path regression tests; no model loading or library generation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "apd_configs", Path(__file__).resolve().parents[1] / "code/dia/configure_apd_libraries.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class BundleBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "pretrained_models/pretrained_models_v3.zip"
        self.base = {"PEPTDEEP_HOME": str(self.root),
                     "local_model_zip_name": "pretrained_models_v3.zip",
                     "model_mgr": {"model_type": "generic"}}

    def test_all_paths_agree(self):
        module.validate_stock_bundle_path(self.base, self.bundle, self.bundle)

    def test_wrong_zip_name_fails(self):
        self.base["local_model_zip_name"] = "pretrained_models.zip"
        with self.assertRaisesRegex(ValueError, "must select"):
            module.validate_stock_bundle_path(self.base, self.bundle, self.bundle)

    def test_same_filename_elsewhere_fails(self):
        with self.assertRaisesRegex(ValueError, "base YAML"):
            module.validate_stock_bundle_path(self.base, self.root / "elsewhere/pretrained_models_v3.zip", self.bundle)

    def test_runtime_ignores_yaml_path_is_detected(self):
        with self.assertRaisesRegex(ValueError, "loader selects"):
            module.validate_stock_bundle_path(self.base, self.bundle, self.root / "other/pretrained_models_v3.zip")

    def test_missing_model_home_fails(self):
        del self.base["PEPTDEEP_HOME"]
        with self.assertRaisesRegex(ValueError, "PEPTDEEP_HOME"):
            module.validate_stock_bundle_path(self.base, self.bundle, self.bundle)

    def test_non_generic_model_fails(self):
        self.base["model_mgr"]["model_type"] = "hla"
        with self.assertRaisesRegex(ValueError, "generic"):
            module.validate_stock_bundle_path(self.base, self.bundle, self.bundle)

    def test_pinned_hash_is_v3_not_unversioned(self):
        self.assertEqual(module.EXPECTED_STOCK_BUNDLE_SHA256,
                         "345cb24154843ad320181b6feaeaf6e7a284d88f1b52dfe9648e0aa5d594b958")

    def test_wrong_package_version_fails_before_import(self):
        with patch.object(module.importlib.metadata, "version", return_value="2.0.0"):
            with self.assertRaisesRegex(ValueError, "frozen peptdeep"):
                module.installed_stock_bundle()


if __name__ == "__main__":
    unittest.main()
