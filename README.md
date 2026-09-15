# Microbial MS/MS model fine-tuning

Code accompanying **Fine-Tuning MS/MS Prediction Models for Microbial
Proteomics**, by Yongqi Ou, Yuqian Gao, Miguel Fuentes-Cabrera and Aivett Bilbao.

This repository contains scripts for microbial DDA model fine-tuning and
evaluation, and for comparing DIA identification workflows. Manuscript files,
figures, result tables, experimental data and model weights are not included.

## Contents

- `code/dda/`: data preparation, training, paired prediction evaluation,
  collision-energy calibration, training-exposure and bootstrap analysis.
- `code/dia/`: library configuration, identification summaries, independent
  summary verification, library-universe and seed checks.
- `environment.yml`: direct dependency versions.
- `environment_lock/`: the archived Python package list.
- `SOURCE_MANIFEST.json`: source-file hashes and documentation changes.
- `tools/`: package integrity and model-path regression checks.

See [USAGE.md](USAGE.md) for the environment, inputs and main scripts.
Citation information is in [CITATION.cff](CITATION.cff).

## Package checks

```bash
python tools/check_code_package.py
python -m unittest discover -s tools -p 'test_*.py'
```

The integrity check uses the Python standard library; the model-path tests
also require PyYAML. These check packaging and model selection, not the full
research workflow.

## License

Original project code is available under [BSD-2-Clause](LICENSE).
External software retains its own terms; see [SOFTWARE.md](SOFTWARE.md).

## Acknowledgments

This work was completed as part of CS 7980 (Research Capstone) and CS 8674
(Master’s Project) at Northeastern University and used the Explorer Cluster,
supported by Northeastern University's Research Computing team. AI tools
assisted with code development and checking. The authors reviewed the analyses
and results.
