#!/bin/bash
#SBATCH --account=oga96
#SBATCH --job-name=baselinev2
#SBATCH --partition=gpu
#SBATCH --constraint=NOPREEMPT
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=gpb06,gpf01
#SBATCH --output=/projects/oga96/cti99/BASELINE_V2/logs/bv2-%x.%N.%j.out
#SBATCH --error=/projects/oga96/cti99/BASELINE_V2/logs/bv2-%x.%N.%j.err

cd /projects/oga96/cti99/BASELINE_V2
module purge
module load gcc/11.2.0
module load miniforge3/25.3.1
source $HOME_MINIFORGE/miniforge.rc
conda activate geo_stream_py312
python3 -c "import torch;print('CUDA',torch.cuda.is_available())"
python3 -u baseline_v2.py
