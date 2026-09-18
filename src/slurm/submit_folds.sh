#!/bin/bash
#SBATCH --account=oga96
#SBATCH --job-name=v2folds
#SBATCH --partition=gpu
#SBATCH --constraint=NOPREEMPT
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=gpb06,gpf01
#SBATCH --array=1-3%2
#SBATCH --output=/projects/oga96/cti99/BASELINE_V2/results_folds/logs/fold-%A_%a.out
#SBATCH --error=/projects/oga96/cti99/BASELINE_V2/results_folds/logs/fold-%A_%a.err

cd /projects/oga96/cti99/BASELINE_V2
module purge
module load gcc/11.2.0
module load miniforge3/25.3.1
source $HOME_MINIFORGE/miniforge.rc
conda activate geo_stream_py312
python3 -u baseline_v2_folds.py --fold $SLURM_ARRAY_TASK_ID
