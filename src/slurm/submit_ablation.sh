#!/bin/bash
#SBATCH --account=oga96
#SBATCH --job-name=ablv2
#SBATCH --partition=gpu
#SBATCH --constraint=NOPREEMPT
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=gpb06,gpf01
#SBATCH --array=0-6%2
#SBATCH --output=/projects/oga96/cti99/BASELINE_V2/ablation/logs/abl-%A_%a.out
#SBATCH --error=/projects/oga96/cti99/BASELINE_V2/ablation/logs/abl-%A_%a.err

CONFIGS=(full chm_only no_dtm no_ortho no_s2 no_chmstats no_aux)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "=== CONFIG: $CFG ==="

cd /projects/oga96/cti99/BASELINE_V2
module purge
module load gcc/11.2.0
module load miniforge3/25.3.1
source $HOME_MINIFORGE/miniforge.rc
conda activate geo_stream_py312
python3 -u ablation_v2.py --config $CFG
