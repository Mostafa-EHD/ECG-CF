# ECG-CF: counterfactual audit of explanation faithfulness in ECG-language models

Code and data for the paper *"ECG-CF: Auditing ECG-Language Model Explanations with Physiological Counterfactuals"* (under submission).

ECG-CF edits a single physiological component of a normal 12-lead ECG (anterior ST elevation,
T-wave inversion, QT prolongation, QRS widening, atrial fibrillation) at graded intensities around
its guideline threshold, verifies every edit by independent re-delineation, and gives the same
counterfactual to ECG-language models as a signal and/or an image. The answers are coded to test
whether the decision changed and whether the explanation cites, localises or denies the edited
component.

## Contents

| Folder | Content |
|---|---|
| `ecgcf/` | Generator and verification: `ecg_cf.py` (edits, delineation, criteria), `ecg_render.py` (printout images), `run_ptbxl.py` (base selection and edits on PTB-XL), `synthetic.py`, `validate_synthetic.py`, `calibrate.py`, `audit.py` (held-out stress audit and AF calibration), `calibration_and_contamination.ipynb` (PLI ROC, 12SL comparison, contamination search of the released training files) |
| `coding/` | Explanation coder: `faith_parse.py` (v5; SHA-256 of `faith_parse.py` and `denial.py` in `coding/data/`), `denial.py` (denial rules; final version v7), `qtc_stated.py` |
| `inference/` | Model queries: ECG-R1 (`main_r1.py`) and GEM (`main_gem.py`) with Slurm scripts; PULSE notebooks (`pulse/`) |
| `external/` | Georgia (G12EC) replication and reciprocal-ST arm |
| `analysis/` | Metrics, statistics and figures: `analyze.py` (per-model tables), `analysis_v5.py`, `stats_v5.py`, `denial_template.py` (denial-template check, used by `analysis_rev3.py`), `analysis_rev3.py` (ST reading, contradiction rate, final denial rules -> results/rev4/), `stats_rev3.py` (confirmatory tests, Holm), `make_fig_*.py` |
| `validation/` | Coder validation: items (`items_v3.csv`), consensus codes, annotation guide (`annotation_guide_fr.md` and `build_items_v3.py`, in French, as given to the annotators), `validate_coder.py` |
| `results/` | All model answers (`*/responses.jsonl`), manifests of the edits, tables of the paper |
| `figures/` | Figures of the paper |

Note: code comments are in English, except in the four notebooks (`ecgcf/calibration_and_contamination.ipynb`, `inference/pulse/*.ipynb`), whose markdown and comments are partly in French.

## Reproducing the tables

All analysis scripts are run from the repository root:

```bash
pip install -r requirements.txt
python analysis/analysis_v5.py        # main metrics  -> results/v5/
python analysis/stats_v5.py           # tests (2,000 resamples)
python analysis/stats_rev3.py         # tests as reported (50,000 resamples) -> results/rev3/stats.csv
python analysis/analysis_rev3.py      # ST reading, contradiction rate -> results/rev4/
python validation/validate_coder.py   # coder validation (Table "Validation of the coder")
```

These steps use only the files in `results/` and do not need the ECG data or a GPU.

## Regenerating the counterfactual ECGs

1. Download PTB-XL v1.0.3 (and PTB-XL+ for the 12SL comparison) from PhysioNet; for the external
   replication, the Georgia set of the PhysioNet/CinC Challenge 2021 v1.0.3 (`external/fetch_georgia.py`).
2. Run `ecgcf/run_ptbxl.py` (paths at the top of the file). Edits are deterministic given the seeds,
   and `results/r1/manifest.csv` lists every edit with its verification outcome.
3. Model inference requires the released weights of ECG-R1-8B-RL, GEM and PULSE-7B and a GPU. The
   scripts in `inference/` and `external/` are written for a Slurm cluster (Jean Zay) and read their
   paths from environment variables; the `.slurm` files set them and must be adapted to your
   cluster (account placeholder `<project>`, `$WORKDIR`, `$SCRATCH`; the repository is expected in
   `$WORKDIR/ecg-cf`):

   | Variable | Content |
   |---|---|
   | `ECGCF_WORK` | output folder (signals, images, `responses.jsonl`) |
   | `PTBXL_DIR` | PTB-XL v1.0.3 |
   | `GEORGIA_DIR` | Georgia (G12EC) data, external replication only |
   | `ECGR1_DIR` | ECG-R1 code and weights |
   | `GEM_DIR`, `GEM_MODEL` | GEM code and locally patched model (`mm_ecg_tower`) |
   | `WORKDIR`, `SCRATCH` | cluster work and scratch spaces used by the Slurm/setup scripts |

   In the released `responses.jsonl`, the `ecg` and `img` fields hold paths relative to `ECGCF_WORK`.

`results/ptbxl_ids_in_training.json` lists the PTB-XL record identifiers found in the released
training files of the three models (contamination search); only recordings absent from it were used
as base recordings.

## Licence

Code: MIT. Model answers, codes and tables: CC BY 4.0. See `LICENSE`.
`results/fig2/1287.npz` contains one PTB-XL recording (ecg_id 1287) and its edits for Figure 2;
PTB-XL (Wagner et al., 2020, *Sci. Data* 7:154) is CC BY 4.0 and must be cited when reusing it. No other PTB-XL
or Georgia signal is redistributed. The data sets themselves remain under their PhysioNet licences.

## Acknowledgements

Computations were performed using HPC resources from GENCI–IDRIS (Jean Zay).
