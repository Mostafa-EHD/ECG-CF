"""ECG-CF review experiments, GEM-7B (signal + image), Jean Zay.
Georgia external replication + PTB-XL reciprocal-ST edit, on the counterfactuals built by
build_ext.py (same items as ECG-R1 and PULSE). Resumes from OUT/responses.jsonl."""
import json, os, sys, time, subprocess
import numpy as np, pandas as pd
T_START = time.time()
MOCK = os.environ.get("PILOT_MOCK") == "1"
WORK = os.environ["ECGCF_WORK"]; GDIR = os.environ["GEORGIA_DIR"]; GEM_DIR = os.environ.get("GEM_DIR", "")
TIME_BUDGET_H = float(os.environ.get("ECGCF_HOURS", 8.5))
MAX_NEW_TOKENS = 1024
PTBXL_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MIMIC_ORDER = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
TO_MIMIC = [PTBXL_ORDER.index(l) for l in MIMIC_ORDER]
OUT = f"{WORK}/ext/gem"
os.makedirs(f"{OUT}/png", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_queue import build_queue, signal_of
queue = [dict(q, modality="both") for q in build_queue(WORK, GDIR, full=False)]
RESP = f"{OUT}/responses.jsonl"
seen = {}
if os.path.exists(RESP):
    for line in open(RESP):
        try:
            r = json.loads(line); seen[r["sample_id"]] = r
        except Exception:
            pass
todo = [q for q in queue if q["sample_id"] not in seen]
print(f"queue: {len(queue)} | done: {len(queue) - len(todo)} | remaining: {len(todo)}", flush=True)
print(pd.DataFrame(queue).groupby(["block", "role"]).size().to_string(), flush=True)
from ecg_render import render
def prepare(it):
    x = np.asarray(signal_of(it, WORK, GDIR), float)
    png = f"{OUT}/png/{it['sample_id']}.png"
    if not os.path.exists(png):
        render(x, 500, png)
    return x[:5000, TO_MIMIC], png

# ---------------- prompt: official GEM grounding prompt ----------------
PROMPT = None
if not MOCK:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("LANSG/ECG-Grounding", "ecg_jsons/ecg-grounding-test-mimiciv.json", repo_type="dataset")
    row = json.load(open(p))[0]
    PROMPT = row["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "").strip()
else:
    PROMPT = "Interpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence."
print("official GEM prompt:", repr(PROMPT), flush=True)

# ---------------- model ----------------
if not MOCK:
    import torch
    sys.path.insert(0, GEM_DIR)
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from PIL import Image
    MODEL = os.environ["GEM_MODEL"]                      # local copy with mm_ecg_tower patched
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL, None, "gem-llava-v1.6-vicuna-7b")
    model.eval()
    # the ECG tower must hold GEM's published weights (not the placeholder checkpoint)
    from safetensors import safe_open
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    params = dict(model.named_parameters())
    keys = [k for k in idx if k.startswith("model.ecg_tower.")]
    bad = []
    for k in keys[:: max(1, len(keys) // 12)]:
        with safe_open(os.path.join(MODEL, idx[k]), "pt") as f:
            ref = f.get_tensor(k).float()
        got = params[k].detach().float().cpu()
        if not torch.allclose(got, ref, atol=1e-2, rtol=1e-2):
            bad.append(k)
    print(f"ECG encoder: {len(keys)} published weights, check {12 - len(bad)}/12 identical", flush=True)
    assert not bad, f"ECG encoder weights differ: {bad[:3]}"


def ask(sig, png):
    if MOCK:
        return "Sinus rhythm. ST elevation in V1-V4. QTc 470 ms.", 0.0
    qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    img = Image.open(png).convert("RGB")
    imt = process_images([img], image_processor, model.config)[0]
    ecg = torch.tensor(np.nan_to_num(sig).T.astype(np.float32)).half()   # 12 x 5000, as in the authors' eval
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(ids, ecgs=ecg, images=imt.unsqueeze(0).half().cuda(), image_sizes=[img.size],
                             do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip(), time.time() - t0


for q in [queue[0], next(q for q in queue if q["role"] == "positive")]:
    txt, dt = ask(*prepare(dict(q)))
    print(f"--- TEST {q['sample_id']} ({dt:.0f}s)\n{txt[:700]}\n", flush=True)
    if len(txt) < 30:
        raise SystemExit("STOP: empty or inconsistent response")

n, t_loop = 0, time.time()
with open(RESP, "a") as f:
    for q in todo:
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
            print("time budget reached: re-run sbatch ext_gem.slurm", flush=True); break
        try:
            txt, dt = ask(*prepare(dict(q)))
        except Exception as e:
            print("skipped:", q["sample_id"], str(e)[:200], flush=True); continue
        r = dict(q); r.update(answer=txt, seconds=dt, model="GEM-7B")
        f.write(json.dumps(r) + "\n"); f.flush(); seen[r["sample_id"]] = r; n += 1
        if n % 100 == 0:
            el = time.time() - t_loop
            print(f"{n}/{len(todo)}  {el/60:.0f} min  {el/n:.1f} s/response  (~{(len(todo)-n)*el/n/3600:.1f} h left)", flush=True)
left = [q for q in queue if q["sample_id"] not in seen]
print("DONE" if not left else f"{len(left)} requests left -> re-run sbatch ext_gem.slurm", flush=True)
subprocess.run(["zip", "-qj", f"{WORK}/ext_gem_resultats.zip", RESP])
print("file to send me:", f"{WORK}/ext_gem_resultats.zip", flush=True)
