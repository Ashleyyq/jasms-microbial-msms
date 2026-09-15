#!/usr/bin/env python3
"""Read-only integrity and boundary checks for the code package."""

import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {"README.md", "USAGE.md", "SOFTWARE.md", "LICENSE", "CITATION.cff", ".gitignore",
             "SOURCE_MANIFEST.json", "tools/check_code_package.py", "tools/test_code_package.py"}
BAD_TEXT = (
    re.compile(r"/(?:Users|home)/[^/<]+/"),
    re.compile("/" + r"scratch/[^/<]+/"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def check():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    allowed = set(manifest["files"]) | DOCUMENTS
    errors = []
    for name in manifest["files"]:
        if not re.fullmatch(r"code/(?:dda|dia)/[A-Za-z0-9_]+\.py", name) and name not in {
            "environment.yml", "environment_lock/pip_freeze_epoch10_exact_run.txt",
            "tools/test_apd_bundle_binding.py",
        }:
            errors.append(f"out_of_scope_manifest_entry:{name}")
    seen = set()
    python_files = 0
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        if path.is_symlink():
            errors.append(f"symlink:{rel}")
            continue
        if not path.is_file():
            continue
        name = rel.as_posix()
        seen.add(name)
        if name not in allowed:
            errors.append(f"unexpected_file:{name}")
        data = path.read_bytes()
        expected = manifest["files"].get(name)
        if expected and hashlib.sha256(data).hexdigest() != expected:
            errors.append(f"changed_source:{name}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"binary:{name}")
            continue
        if any(pattern.search(text) for pattern in BAD_TEXT):
            errors.append(f"private_path_or_secret_marker:{name}")
        if path.suffix == ".py":
            try:
                compile(ast.parse(text), name, "exec")
                python_files += 1
            except SyntaxError:
                errors.append(f"syntax:{name}")
    errors += [f"missing:{name}" for name in sorted(allowed - seen)]
    return {"integrity_passed": not errors, "files": len(seen),
            "python_files_compiled_not_executed": python_files, "errors": errors,
            "scope": "File integrity and packaging only; not a license or publication decision"}


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["integrity_passed"] else 1)
