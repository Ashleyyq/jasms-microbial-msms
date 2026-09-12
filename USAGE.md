# Using the code

## Environment

The archived DDA environment used Linux, Python 3.9, AlphaPeptDeep 1.4.1 and
alphabase 1.8.1. If using Conda, create an environment from the supplied file:

```bash
conda env create -f environment.yml
conda activate bacterial-apd-jasms
```

The archived package list records PyTorch 2.7.1+cu118. The portable environment
file pins 2.7.1 without selecting a CUDA wheel; match CUDA installation to the
target machine before attempting training. No packages are bundled here.

## DDA entry points

| Step | Script |
|---|---|
| Source-group split | `code/dda/make_species_split.py` |
| Remove test-sequence overlap from training | `code/dda/make_trainfold_clean_seqs.py` |
| Train candidate epoch checkpoints | `code/dda/t3_finetune.py` |
| Paired spectrum evaluation | `code/dda/eval_paired_full.py` |
| Pretrained NCE sweep | `code/dda/eval_stock_nce_sweep.py` |
| Training-exposure analysis | `code/dda/stratify_external_seen_unseen.py` |

Public development and testing inputs come from PXD010000 and PXD010613.
Obtain and convert the inputs before running these entry points. Fold lists,
the cleaned training sequence list, models and output locations must be supplied
as appropriate to each script. The trained epoch-10 binary is not distributed.

Other DDA files include shared functions and earlier comparison utilities.
In particular, `run_all.py` supplies helper functions used by later scripts;
its historical all-in-one command is not the final manuscript workflow.

## DIA entry points

`code/dia/configure_apd_libraries.py` builds paired AlphaPeptDeep configurations
from a user-supplied base YAML. It checks the v3 pretrained bundle identity and
path before writing configurations. `summarize_pandas.py` and
`summarize_pyarrow.py` provide independent summary implementations;
`verify_summaries.py` compares their outputs. The remaining utilities check
seed comparisons and the exported library universe.

The study used P1 for Carafe adaptation and P2/P3 for held-out evaluation.
Seed 2024 was primary, with 2025 and 2026 as sensitivity analyses. Carafe and
DIA-NN are external software; obtain the relevant versions under terms
applicable to the intended use and compute environment. This repository does
not include their executables, models, report files or cluster job wrappers.

Running the analyses requires external data, split lists, configurations and
models in addition to this code.
