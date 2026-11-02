#!/bin/bash
#SBATCH --job-name=ecgcf_ext_r1
#SBATCH --output=logs/ext_r1_%j.out
#SBATCH --error=logs/ext_r1_%j.err
#SBATCH -C a100
#SBATCH --account=<project>@a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
BASE=$WORKDIR
module purge; module load python/3.11.5 2>/dev/null || module load python
source $BASE/env/bin/activate
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ECGCF_WORK=$SCRATCH/ecgcf_out ECGR1_DIR=$BASE/ECG-R1 GEORGIA_DIR=$SCRATCH/georgia
export ECGCF_BATCH=16 ECGCF_DTYPE=bfloat16 ECGCF_HOURS=7.5
export XDG_CACHE_HOME=$SCRATCH/.cache TRITON_CACHE_DIR=$SCRATCH/.triton
cd $SCRATCH/ext_run
python -u main_ext_r1.py
