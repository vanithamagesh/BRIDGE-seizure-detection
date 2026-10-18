# BRIDGE: Patient-Baseline-Referenced Dual-Graph Learning for Cross-Patient Seizure Detection

Code for the conference paper *"BRIDGE: Patient-Baseline-Referenced Dual-Graph Learning for Cross-Patient Seizure Detection on Edge Devices"* (authors and venue to be added).

BRIDGE is a graph neural network of 7.1k parameters for seizure detection in scalp EEG. Each electrode channel is a graph node that carries its spectral and temporal features twice: as measured, and relative to the same patient's causal running median over the previous 10 minutes. Message passing runs over a learned mix of an anatomical electrode graph and the correlation graph of each 4-s window. Decision thresholds are calibrated on training subjects only, under a false-alarm budget, and uncertain windows can be referred for review.

## Main results (CHB-MIT, leave-one-subject-out, 23 subjects)

| Model | AUROC | AUPRC | Seizures detected (%) | False alarms / h |
|---|---|---|---|---|
| BRIDGE | 0.944 | 0.599 | 72.2 | 0.96 |
| BRIDGE-GT | 0.928 | 0.598 | 71.2 | 0.79 |
| Random forest | 0.942 | 0.614 | 71.3 | 0.87 |
| Transformer (no graph) | 0.919 | 0.614 | 80.4 | 1.23 |
| Raw 1-D CNN | 0.818 | 0.281 | 38.4 | 5.16 |

On the fixed patient split used by BIOT, CBraMod and CSBrain (train chb01–19, validate chb20–21, test chb22–23), BRIDGE reaches an AUROC of 0.990 ± 0.001 over three seeds. The int8 model occupies 20 kB and classifies one window in 1.3 ms on one CPU thread. All numbers are in `results/`.

## Repository

| Path | Content |
|---|---|
| `bridge_chbmit.py` | The whole study in one script: download, features, patient-baseline reference, BRIDGE, BRIDGE-GT and baselines, leave-one-subject-out evaluation, calibration and abstention, ablations, leakage check, int8 quantization, fixed-split comparison, figures and tables |
| `figures/feat_local.py` | Builds `features_local.npz`, which the figure script uses for the correlation maps |
| `figures/make_paper_figs.py` | Redraws the result figures of the paper from a results folder, without retraining |
| `figures/make_fig_isolating_1col.py` | Draws the isolating-ablation figure of the paper |
| `results/loso_fixed_split/` | Results of the full run used in the paper (per subject, summary, ablations, efficiency, fixed split) |
| `results/isolating_ablations/` | Results of the isolating ablations used in the paper |

## Data

The CHB-MIT Scalp EEG Database v1.0.0 (Shoeb 2009; Goldberger et al., PhysioNet, 2000) is open access under ODC-BY 1.0. The script downloads it from the PhysioNet AWS open-data mirror, falling back to physionet.org. Nothing has to be downloaded by hand.

## How to run

**Google Colab (recommended).** Runtime → Change runtime type → GPU (A100 or T4). Upload `bridge_chbmit.py`, then run:

```
!pip install -r requirements.txt      # Colab already has these packages
!python bridge_chbmit.py --smoke      # about 5 min: checks the whole pipeline on 3 cases
!python bridge_chbmit.py              # full run: about 7.5 GB download, 1.5-2 h on an A100
!python bridge_chbmit.py --isolate    # isolating ablations, about 30 min (reuses the feature cache)
```

**Local machine** (Python 3.10+, a CUDA GPU recommended):

```
pip install -r requirements.txt
python bridge_chbmit.py --data ./chbmit --out ./chbmit_out
```

If a run stops, start it again with the same command. Finished subjects, ablations and seeds are loaded from checkpoints. The results are written to the output folder and packed into `bridge_results.zip`.

**Paper figures from the stored results:**

```
python figures/feat_local.py ./edf_cache ./features
python figures/make_paper_figs.py results/loso_fixed_split ./features/features_local.npz ./edf_cache ./paper_figures
python figures/make_fig_isolating_1col.py results/isolating_ablations/isolate_summary.json ./paper_figures
```

| Paper figure | File | Made by |
|---|---|---|
| Fig. 1, architecture | `fig1_architecture` | `bridge_chbmit.py` (full run) |
| Fig. 2, baseline reference | `fig11_baseline_reference_1col` | `make_paper_figs.py` |
| Fig. 3, AUROC per subject | `fig2b_per_subject_auroc_1col` | `make_paper_figs.py` |
| Fig. 4, ablation and leakage | `fig5_ablation_leakage_1col` | `make_paper_figs.py` |
| Fig. 5, isolating ablations | `fig12_isolating_ablations_1col` | `make_fig_isolating_1col.py` |
| Fig. 6, example recording | `fig7_example_timeline_1col` | `make_paper_figs.py` |
| Fig. 7, efficiency | `fig6_efficiency` | `make_paper_figs.py` |
| Fig. 8, learned graphs | `fig8_graphs_1col` | `make_paper_figs.py` |

## Evaluation protocol in brief

- **Leave-one-subject-out:** 23 subjects; chb21 is a later recording of the subject of chb01 and is merged with it. Each case contributes its seizure files plus one seizure-free file.
- **Calibration:** thresholds (maximum F1; at most one false alarm per hour) and abstention thresholds come from three inner subject-wise folds of the training subjects only.
- **Alarms:** an alarm is raised when two of the last three 4-s windows are positive. A false alarm is an alarm run that does not overlap a seizure (plus 30 s).
- **Fixed split:** as in BIOT, CBraMod and CSBrain. The threshold comes from the validation cases, and results are reported over three seeds.

## Citation

Please cite the paper (reference to be added after acceptance) and the CHB-MIT database:

- A. H. Shoeb, "Application of machine learning to epileptic seizure onset detection and treatment," Ph.D. dissertation, MIT, 2009.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.

## License

MIT License (see `LICENSE`). The CHB-MIT data keep their own license (ODC-BY 1.0).
