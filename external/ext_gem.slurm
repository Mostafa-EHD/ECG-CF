#!/bin/bash
#SBATCH --job-name=ecgcf_ext_gem
#SBATCH --output=logs/ext_gem_%j.out
#SBATCH --error=logs/ext_gem_%j.err
#SBATCH -C a100
#SBATCH --account=<project>@a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=09:00:00
S=$SCRATCH/gem
module purge; module load python/3.11.5 2>/dev/null || module load python
source $S/env/bin/activate
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=$S/.cache
export ECGCF_WORK=$SCRATCH/ecgcf_out GEORGIA_DIR=$SCRATCH/georgia GEM_DIR=$S/GEM GEM_MODEL=$S/gem_model ECGCF_HOURS=8.5
export PYTHONPATH=$S/GEM:$PYTHONPATH
cd $SCRATCH/ext_run
python -u main_ext_gem.py
