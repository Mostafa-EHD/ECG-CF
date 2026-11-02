#!/bin/bash
# ECG-CF on Jean Zay: installation (run ONCE, on the login node, which has Internet).
# The compute nodes have no Internet: everything is downloaded here in advance.
set -e
BASE=$WORKDIR
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $BASE && cd $BASE
export HF_HOME=$BASE/hf_cache PIP_CACHE_DIR=$BASE/pip_cache TORCH_HOME=$BASE/.torch_cache

# 1. Python environment (same versions as the Kaggle pilot, tested)
module purge; module load python/3.11.5 2>/dev/null || module load python
python -m venv $BASE/env
source $BASE/env/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
[ -d ECG-R1 ] || git clone -q --depth 1 https://github.com/PKUDigitalHealth/ECG-R1.git
pip install -q "transformers==4.57.1" accelerate addict attrdict binpacking dacite "datasets>=3.0,<4.0" einops \
  importlib_metadata jieba json_repair modelscope nltk "peft>=0.11,<0.18" rouge sentencepiece simplejson \
  sortedcontainers tiktoken transformers_stream_generator "trl>=0.15,<0.24" zstandard "qwen_vl_utils==0.0.14" \
  ftfy timm wfdb neurokit2 cpm_kernels oss2 openai huggingface_hub safetensors scipy pandas matplotlib
pip install -q --no-deps -e ECG-R1

# 2. models and prompt files (local Hugging Face cache)
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("PKUDigitalHealth/ECG-R1-8B-RL")
snapshot_download("ncbi/MedCPT-Query-Encoder")
for f in ["image_missing100", "ecg_missing100", "full"]:
    hf_hub_download("PKUDigitalHealth/ECG-Protocol-Guided-Grounding-CoT",
                    f"ecg_jsons/test_set/ecg-grounding-test-mimiciv_{f}.jsonl", repo_type="dataset")
print("models downloaded")
PY

# 3. PTB-XL 1.0.3 (if the download is blocked: copy it from your machine with rsync, see LISEZMOI)
if [ ! -f $BASE/ptb-xl/ptbxl_database.csv ]; then
  wget -q -O ptbxl.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip \
    && unzip -q ptbxl.zip -d ptbxl_tmp && mv ptbxl_tmp/*/ $BASE/ptb-xl && rm -rf ptbxl.zip ptbxl_tmp
fi
ls $BASE/ptb-xl/ptbxl_database.csv && echo "PTB-XL OK"

# 4. experiment code
mkdir -p $BASE/run && cp $HERE/*.py $HERE/*.json $HERE/job.slurm $BASE/run/
echo "Installation complete. Run: cd $BASE/run && sbatch job.slurm"
