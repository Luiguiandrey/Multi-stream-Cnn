# CNN vs DRF for French NFI forest-attribute prediction

Code and reference data for the manuscript:
**"[Multi-Stream Deep Learning Outperforms Distributional Random Forests for National Forest Inventory Attribute Prediction: A Benchmark Across Ecoregions and Photogrammetric Systems ]"**, [Luigui Andrey Ramirez Parra, Cedric Vega, Jean Pierre Renaud, Antoine Labatie], *submitted to International Journal of Applied Earth Observation and Geoinformation*, [2026].

## Repository structure
```
src/                 model, spatial CV, ablation, quantile-regression code
  slurm/             SLURM job submission templates
data/                anonymised reference labels + split definitions (see data/README.md)
requirements.txt     Python package versions
CITATION.cff         citation metadata
```

## Target attributes
Volume (v_wac), basal area (ba_wac), quadratic mean diameter (qmd_wac),
dominant height (H0_wac), GSVI (pv_wac).

## Environment
Python 3.12, PyTorch 2.6.0 (CUDA 12.4), timm 1.0.27. Full list in `requirements.txt`.
Trained on NVIDIA P100 (16 GB) GPUs under a SLURM-managed cluster.
```
pip install -r requirements.txt
```

## Run order
1. `src/baseline_v2.py` - main multi-stream CNN (full model, 90/10 split)
2. `src/baseline_v2_folds.py --fold {1,2,3}` - spatial cross-validation
3. `src/ablation_v2.py --config {...}` - input-stream ablation
4. `src/baseline_v2_quantile.py` - quantile-regression variant (predictive intervals)

SLURM templates for each are in `src/slurm/`.

**Note on identifiers:** published data files use anonymised `plot_id` values. The
training scripts expect the original NFI `npp` identifier, which is provided together
with the imagery on request.

## Data availability
**Code.** All scripts are released under the MIT License.

**Reference labels.** `data/nfi_minimal.csv` contains only an anonymised plot identifier
and the five target attributes; forest-type and altitude classes are in the split files.
No coordinates, species-level breakdowns, or other inventory variables are included.
Split definitions (`data/calval_used.csv`, `data/test_used.csv`) reproduce the exact
partitioning, including spatial folds.

**Remote-sensing patches.** The image inputs (CHM, DTM, orthophoto RGB/NIR, and
Sentinel-2 patches) are not redistributed here. They are derived from data provided by
the French National Institute of Geographic and Forest Information (IGN) and are subject
to its data-use terms. These data are available from the authors on reasonable request
via cloud-storage transfer, following removal of identifying metadata and re-coding of
plot identifiers, in accordance with IGN's data-sharing conditions. Requests should be
directed to [CORRESPONDING AUTHOR EMAIL].

## Citation
[Ramirez-Parra, L. A., Vega, C., Renaud, J.P., & Labatie, A. (2026). Multi-stream CNN for French NFI forest-attribute prediction (v1.0.0) [Pythorch lighting ].]

Zenodo: [] | Repository: https://github.com/Luiguiandrey/Multi-stream-Cnn

## Acknowledgements
This work received government funding managed by the Agence Nationale de la Recherche under the France 2030 program as part of the "Forest Resilience" research program (PEPR FORESTT), reference number  ANR-24-PEFO-0003.
This work was supported by the interdisciplinary program ARTEMIS of Lorraine Université d’Excellence (ANR-15-IDEX-04-LUE)
This work was supported by the TOSCA CNES projects  CFOREST-50M grant number 580000431 and FORGE3D 5800004319

