#!/bin/bash
#SBATCH --job-name=ecgcf_build
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err
#SBATCH -C a100
#SBATCH --account=<project>@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
BASE=$WORKDIR
module purge; module load python/3.11.5 2>/dev/null || module load python
source $BASE/env/bin/activate
export ECGCF_WORK=$SCRATCH/ecgcf_out GEORGIA_DIR=$SCRATCH/georgia PTBXL_DIR=$SCRATCH/ptb-xl N_BASE=150
cd $SCRATCH/ext_run
python -u build_ext.py
