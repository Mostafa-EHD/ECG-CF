#!/bin/bash
#SBATCH --job-name=ecgcf_r1_pos
#SBATCH --output=logs/ecgcf_%j.out
#SBATCH --error=logs/ecgcf_%j.err
#SBATCH -C a100                      # GPU A100 80 GB (bfloat16); for H100: -C h100
#SBATCH --account=<project>@a100           # check with the command: idracct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --qos=qos_gpu_a100-t3
BASE=$WORKDIR
module purge; module load python/3.11.5 2>/dev/null || module load python
source $BASE/env/bin/activate
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ECGCF_WORK=$SCRATCH/ecgcf_out ECGR1_DIR=$BASE/ECG-R1 PTBXL_DIR=$SCRATCH/ptb-xl
export ECGCF_BATCH=16 ECGCF_DTYPE=bfloat16 ECGCF_HOURS=2.5
mkdir -p logs $ECGCF_WORK
cd $WORKDIR/ecg-cf/inference; export XDG_CACHE_HOME=$SCRATCH/.cache TRITON_CACHE_DIR=$SCRATCH/.triton
EXTRA_POS=1 python -u main_r1.py
