# PV Defect Screening Under Acquisition Shift: Robustness and Calibration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23537392.svg)](https://doi.org/10.5281/zenodo.23537392)

This is the follow-up to my published ELPV benchmark ([pv-el-defect-screening](https://github.com/KhandokerSamiulHoque/pv-el-defect-screening), IJSRA 2026, [doi:10.30574/ijsra.2026.20.2.1601](https://doi.org/10.30574/ijsra.2026.20.2.1601)). It asks what happens to that screening model when electroluminescence images are captured differently from the training set. Exposure, contrast, noise and focus all vary between labs and cameras.

## What it does
- Trains the same interpretable logistic model as the published paper (386 features: 62 intensity and gradient features plus 324 HOG features) on the original ELPV images only.
- Scores the held-out folds under six controlled conditions: reference, underexposure (x0.70), overexposure (x1.15 + 20), low contrast (65%), Gaussian noise (sigma 12) and Gaussian blur (radius 1.25).
- Reports accuracy, F1, ROC-AUC, PR-AUC, Brier score, 10-bin expected calibration error (ECE), and a simple triage rule that sends scores between 0.40 and 0.60 to a human reviewer.

## Results (2,624 cells, stratified 5-fold, seed 20260727)

| Condition | F1 | ROC-AUC | Brier | ECE | Change in F1 |
|---|---:|---:|---:|---:|---:|
| Reference | 0.673 | 0.785 | 0.192 | 0.087 | 0 |
| Underexposure | 0.674 | 0.774 | 0.224 | 0.177 | +0.001 |
| Overexposure | 0.613 | 0.775 | 0.201 | 0.115 | -0.061 |
| Low contrast | 0.640 | 0.759 | 0.202 | 0.086 | -0.033 |
| Gaussian noise | 0.640 | 0.750 | 0.215 | 0.117 | -0.033 |
| Gaussian blur | 0.667 | 0.780 | 0.196 | 0.089 | -0.006 |

Two findings matter for practice. Overexposure costs the most F1 (0.061), mostly through lost recall. Underexposure keeps F1 almost unchanged but doubles the calibration error (ECE 0.087 to 0.177), so the scores look as confident as before while being less trustworthy. Per-technology results are in `results/published/pv_extension_subgroup_results.csv`. Under reference conditions F1 is 0.713 for monocrystalline and 0.643 for polycrystalline cells.

These reference numbers differ slightly from the published paper (F1 0.670; 0.712 mono and 0.639 poly) because this study uses a different fold seed (20260727 instead of 42). The model and features are the same.

## Reproduce
```bash
pip install -r requirements.txt
git clone https://github.com/zae-bayern/elpv-dataset
export ELPV_DATA=elpv-dataset/src/elpv_dataset/data   # Windows: set ELPV_DATA=...
cd src && python run_pv_extension_experiment.py
```
Outputs go to `results/run/`. `results/published/` holds the reference results and figures. A rerun on 2026-10-20 matched them exactly (maximum absolute difference 0.0 in both result tables).

## Limits
- The shifts are synthetic stress tests, not a second dataset or a second camera.
- The ELPV labels have no module IDs, so the folds are not module-independent.
- Labels are expert defect probabilities, not measured power loss.
- The 0.40 to 0.60 review band is illustrative and was not tuned to a real cost of missed defects.
- Public data only. No employer data, equipment or procedures were used.

## Data license
ELPV images are CC BY-NC-SA 4.0 by their authors and are not redistributed here.

Author: Khandoker Samiul Hoque (ORCID [0009-0008-0756-2395](https://orcid.org/0009-0008-0756-2395)). Code license: MIT. Manuscript in preparation.

## Archived release
Version 1.0.0 is archived on Zenodo: [doi:10.5281/zenodo.23537393](https://doi.org/10.5281/zenodo.23537393). To cite the latest version instead, use [doi:10.5281/zenodo.23537392](https://doi.org/10.5281/zenodo.23537392).
