#!/usr/bin/env python3
"""Write paired stock and epoch-10 AlphaPeptDeep library configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import yaml


EXPECTED_EPOCH10_SHA256 = (
    "a7f36aab17516a7d31acd4c68953e053941b5f25d881d21a21a0133b4d61a008"
)
EXPECTED_STOCK_BUNDLE_SHA256 = (
    "345cb24154843ad320181b6feaeaf6e7a284d88f1b52dfe9648e0aa5d594b958"
)
EXPECTED_STOCK_BUNDLE_NAME = "pretrained_models_v3.zip"


def validate_stock_bundle_path(base: dict, selected: Path, runtime_bundle: Path) -> None:
    """The checked file must be the bundle selected by YAML and the APD loader."""
    if base.get("local_model_zip_name") != EXPECTED_STOCK_BUNDLE_NAME:
        raise ValueError("base settings must select pretrained_models_v3.zip")
    model_home = base.get("PEPTDEEP_HOME")
    if not isinstance(model_home, str) or not model_home.strip():
        raise ValueError("base settings must define PEPTDEEP_HOME")
    if base.get("model_mgr", {}).get("model_type") != "generic":
        raise ValueError("the controlled comparison requires the generic v3 models")
    configured = (Path(model_home).expanduser() / "pretrained_models" /
                  EXPECTED_STOCK_BUNDLE_NAME).resolve()
    selected = selected.expanduser().resolve()
    if configured != selected:
        raise ValueError("--stock-model-bundle is not the bundle selected by the base YAML")
    if Path(runtime_bundle).expanduser().resolve() != selected:
        raise ValueError("the installed APD loader selects a different model bundle")


def installed_stock_bundle() -> Path:
    """Read the loader setting without constructing a model or running inference."""
    if importlib.metadata.version("peptdeep") != "1.4.1":
        raise ValueError("generate the configurations in the frozen peptdeep 1.4.1 environment")
    from peptdeep.pretrained_models import MODEL_ZIP_FILE_PATH
    return Path(MODEL_ZIP_FILE_PATH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(base: dict, fasta: Path, model: str, output: Path) -> dict:
    settings = copy.deepcopy(base)
    settings["thread_num"] = 16
    settings["torch_device"]["device_type"] = "gpu"
    settings["task_workflow"] = ["library"]

    manager = settings["model_mgr"]
    manager["default_nce"] = 25.0
    manager["default_instrument"] = "Lumos"
    manager["external_ms2_model"] = model
    manager["external_rt_model"] = ""

    library = settings["library"]
    library["infile_type"] = "fasta"
    library["infiles"] = [str(fasta)]
    library["fasta"]["protease"] = "trypsin_not_P"
    library["fasta"]["max_miss_cleave"] = 1
    library["fasta"]["add_contaminants"] = False
    library["fix_mods"] = ["Carbamidomethyl@C"]
    library["var_mods"] = ["Oxidation@M"]
    library["min_var_mod_num"] = 0
    library["max_var_mod_num"] = 1
    library["min_precursor_charge"] = 2
    library["max_precursor_charge"] = 4
    library["min_peptide_len"] = 7
    library["max_peptide_len"] = 30
    library["min_precursor_mz"] = 380.0
    library["max_precursor_mz"] = 980.0
    library["decoy"] = "None"
    library["frag_types"] = ["b", "y"]
    library["max_frag_charge"] = 2
    library["output_folder"] = str(output)

    output_tsv = library["output_tsv"]
    output_tsv["enabled"] = True
    output_tsv["min_fragment_mz"] = 150.0
    output_tsv["max_fragment_mz"] = 2000.0
    output_tsv["keep_higest_k_peaks"] = 20
    output_tsv["translate_mod_to_unimod_id"] = True
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-settings", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--stock-model-bundle", type=Path, required=True)
    parser.add_argument("--epoch10-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.base_settings,
        args.fasta,
        args.stock_model_bundle,
        args.epoch10_checkpoint,
    ):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")

    checkpoint_hash = sha256(args.epoch10_checkpoint)
    if checkpoint_hash != EXPECTED_EPOCH10_SHA256:
        raise SystemExit(
            "epoch-10 checkpoint hash mismatch: "
            f"expected {EXPECTED_EPOCH10_SHA256}, got {checkpoint_hash}"
        )
    stock_hash = sha256(args.stock_model_bundle)
    if stock_hash != EXPECTED_STOCK_BUNDLE_SHA256:
        raise SystemExit(
            "stock model bundle hash mismatch: "
            f"expected {EXPECTED_STOCK_BUNDLE_SHA256}, got {stock_hash}"
        )

    with args.base_settings.open() as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise SystemExit("base settings must be a YAML mapping")
    runtime_bundle = installed_stock_bundle()
    validate_stock_bundle_path(base, args.stock_model_bundle, runtime_bundle)

    config_dir = args.run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    definitions = {
        "stock": ("", args.run_root / "libraries" / "apd_stock"),
        "epoch10": (
            str(args.epoch10_checkpoint),
            args.run_root / "libraries" / "apd_epoch10",
        ),
    }
    configs = {}
    for label, (model, output) in definitions.items():
        config_path = config_dir / f"apd_{label}.yaml"
        with config_path.open("x") as handle:
            yaml.safe_dump(
                build_config(base, args.fasta, model, output),
                handle,
                sort_keys=False,
            )
        configs[label] = {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "external_ms2_model": model or "stock-installed-v3",
            "external_rt_model": "stock-installed-v3",
            "output_folder": str(output),
        }

    manifest = {
        "design": (
            "Stock and epoch-10 libraries use identical FASTA digestion, modification, "
            "charge, m/z, NCE, RT model, and TSV export settings. Only the external "
            "MS2 checkpoint differs."
        ),
        "epoch10_checkpoint": str(args.epoch10_checkpoint),
        "epoch10_checkpoint_sha256": checkpoint_hash,
        "stock_model_bundle": str(args.stock_model_bundle),
        "stock_model_bundle_sha256": stock_hash,
        "stock_model_bundle_runtime": str(runtime_bundle.expanduser().resolve()),
        "stock_model_bundle_yaml_and_runtime_verified": True,
        "base_settings": str(args.base_settings),
        "base_settings_sha256": sha256(args.base_settings),
        "fasta": str(args.fasta),
        "fasta_sha256": sha256(args.fasta),
        "configs": configs,
    }
    manifest_path = args.run_root / "apd_library_config_manifest.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
