# Reference data

Plot identifiers are anonymised (`plot_id`, e.g. plot_00001). The original NFI
identifiers used by the training scripts are provided together with the imagery
on request (see the data-availability statement in the top-level README).

## nfi_minimal.csv  (27,433 plots used in the study)
plot_id, v_wac (volume m3/ha), ba_wac (basal area m2/ha), qmd_wac (quadratic mean
diameter m), H0_wac (dominant height m), pv_wac (GSVI m3/ha/yr).
No coordinates, species breakdowns, or other inventory variables.

## calval_used.csv  (25,091 calibration/validation plots)
plot_id, fold (spatial fold 1-3), matching (acquisition system), Ftype
(forest type: D/C/M), Altitude (Faible/Moyen/Elevé).

## test_used.csv  (3,956 independent test plots)
plot_id, Ftype, Altitude. (34 plots lack the derived factors and are excluded
from factor-stratified analyses.)

## Remote-sensing patches
Not included; available on request (see top-level README).
