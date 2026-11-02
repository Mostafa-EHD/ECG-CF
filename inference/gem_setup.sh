#!/bin/bash
# GEM on Jean Zay: installation (ONCE, on the login node). Everything goes in $SCRATCH (WORK quota full).
set -e
S=$SCRATCH/gem
mkdir -p $S && cd $S
export HF_HOME=$S/hf_cache PIP_CACHE_DIR=$S/pip_cache XDG_CACHE_HOME=$S/.cache
module purge; module load python/3.11.5 2>/dev/null || module load python
[ -d env ] || python -m venv env
source env/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
[ -d GEM ] || git clone -q --depth 1 https://github.com/lanxiang1017/GEM.git
pip install -q --no-deps -e GEM
pip install -q "transformers==4.37.2" "tokenizers==0.15.1" "accelerate==0.28.0" "sentencepiece==0.1.99" \
  shortuuid "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13" ftfy regex scipy scikit-learn wfdb \
  neurokit2 pandas matplotlib safetensors huggingface_hub protobuf peft
echo "== environment OK"

python - <<'PY'
import os, json, glob, torch
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors import safe_open
S = os.environ["SCRATCH"] + "/gem"
gem = snapshot_download("LANSG/GEM")
snapshot_download("openai/clip-vit-large-patch14-336")
snapshot_download("ncbi/MedCPT-Query-Encoder")
hf_hub_download("LANSG/ECG-Grounding", "ecg_jsons/ecg-grounding-test-mimiciv.json", repo_type="dataset")
print("== models downloaded")
# checkpoint of the ECG encoder rebuilt from GEM's own published weights (same weights GEM uses)
idx = json.load(open(f"{gem}/model.safetensors.index.json"))["weight_map"]
pre = "model.ecg_tower.ecg_tower."
sd = {}
for f in sorted(set(v for k, v in idx.items() if k.startswith(pre))):
    with safe_open(f"{gem}/{f}", "pt") as h:
        for k in h.keys():
            if k.startswith(pre):
                sd["module.ecg." + k[len(pre):]] = h.get_tensor(k).float()
os.makedirs(f"{S}/ckpt", exist_ok=True)
torch.save({"state_dict": sd}, f"{S}/ckpt/cpt_wfep_epoch_20.pt")
print("== ECG encoder:", len(sd), "tensors")
# local model folder = links to the snapshot + config with the encoder path
m = f"{S}/gem_model"
os.makedirs(m, exist_ok=True)
for p in glob.glob(f"{gem}/*"):
    d = f"{m}/{os.path.basename(p)}"
    if os.path.basename(p) != "config.json" and not os.path.exists(d):
        os.symlink(os.path.realpath(p), d)
cfg = json.load(open(f"{gem}/config.json"))
cfg["mm_ecg_tower"] = f"{S}/ckpt/cpt_wfep_epoch_20.pt"
json.dump(cfg, open(f"{m}/config.json", "w"), indent=2)
print("== model ready:", m)
PY
echo "GEM installation complete. Run: sbatch gem_job.slurm"
