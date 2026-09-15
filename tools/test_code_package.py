"""Check package boundaries in disposable copies, without running analyses."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_check", ROOT / "tools/check_code_package.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "package"
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    def check_package(self):
        with patch.object(CHECKER, "ROOT", self.root):
            return CHECKER.check()

    def test_clean_package(self):
        self.assertTrue(self.check_package()["integrity_passed"])

    def test_changed_source(self):
        path = self.root / "code/dia/verify_summaries.py"
        path.write_text(path.read_text() + "\n# Test-only edit\n")
        self.assertIn("changed_source:code/dia/verify_summaries.py", self.check_package()["errors"])

    def test_missing_citation(self):
        (self.root / "CITATION.cff").unlink()
        self.assertIn("missing:CITATION.cff", self.check_package()["errors"])

    def test_unlisted_file(self):
        (self.root / "unlisted.txt").write_text("Test fixture\n")
        self.assertIn("unexpected_file:unlisted.txt", self.check_package()["errors"])

    def test_symlink(self):
        (self.root / "linked.txt").symlink_to(self.root / "README.md")
        self.assertIn("symlink:linked.txt", self.check_package()["errors"])


if __name__ == "__main__":
    unittest.main()
