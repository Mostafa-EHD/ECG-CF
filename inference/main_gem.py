"""ECG-CF main experiment, GEM-7B (signal + image), Jean Zay.
Reuses the verified counterfactuals produced by main_r1.py ($ECGCF_WORK/main_r1/cf), so GEM
sees exactly the same edited ECGs as ECG-R1 and PULSE. Resumes from OUT/responses.jsonl."""
import ast
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

T_START = time.time()
MOCK = os.environ.get("PILOT_MOCK") == "1"
WORK = os.environ["ECGCF_WORK"]
GEM_DIR = os.environ.get("GEM_DIR", "")
PTBXL = os.environ["PTBXL_DIR"]
TIME_BUDGET_H = float(os.environ.get("ECGCF_HOURS", 9.0))
MAX_NEW_TOKENS = 1024
SUPRA = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3}
CONTROLS = [("ctrl_amp", 2), ("ctrl_noise", 2), ("ctrl_wander", 2)]
POS_CODES = {"st_elev": ["STE_"], "t_inv": ["INVT"], "qt_long": ["LNGQT"],
             "qrs_wide": ["CLBBB", "CRBBB"], "af": ["AFIB"]}
N_POS_PER_CLASS = 20
PTBXL_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MIMIC_ORDER = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
TO_MIMIC = [PTBXL_ORDER.index(l) for l in MIMIC_ORDER]

CF = f"{WORK}/main_r1/cf"
OUT = f"{WORK}/main_gem"
for d in ["wfdb", "png"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
assert os.path.exists(f"{CF}/manifest.csv"), "run main_r1.py first (edited ECGs)"
man = pd.read_csv(f"{CF}/manifest.csv")
base = pd.read_csv(f"{CF}/base_ecgs.csv")
TRAIN_IDS = set(json.load(open("ptbxl_ids_in_training.json")))
if MOCK:
    TRAIN_IDS = {1, 2}
db = pd.read_csv(os.path.join(PTBXL, "ptbxl_database.csv"), index_col="ecg_id")

# ---------------- queue (same base ECGs and order as ECG-R1, block A) ----------------
inc = base[base.excluded.fillna("") == ""].ecg_id.astype(int)
unseen = [e for e in inc if e not in TRAIN_IDS]
order = np.random.default_rng(2026).permutation(sorted(unseen)).tolist()
KEEP = man.set_index("sample_id").keep.to_dict()
queue = []
for e in order:
    queue.append(dict(sample_id=f"{e}_original", ecg_id=int(e), kind="original", level_idx=0, role="original"))
    for k, li in list(SUPRA.items()) + CONTROLS:
        sid = f"{e}_{k}_L{li}"
        if KEEP.get(sid, False):
            queue.append(dict(sample_id=sid, ecg_id=int(e), kind=k, level_idx=li,
                              role="control" if k.startswith("ctrl") else "edit"))
for q in queue:
    q["block"] = "A"
codes = db.scp_codes.apply(ast.literal_eval)
for kind, cl in POS_CODES.items():
    sel = db[codes.apply(lambda c: any(k in c for k in cl)) & (db.strat_fold == 10)
             & (db.validated_by_human == True) & ~db.index.isin(TRAIN_IDS)]  # noqa: E712
    sel = sel.sample(min(N_POS_PER_CLASS, len(sel)), random_state=0)       # same draw as ECG-R1
    queue += [dict(sample_id=f"pos_{kind}_{i}", ecg_id=int(i), kind=kind, level_idx=-1, role="positive",
                   block="B", scp=",".join(codes[i].keys()), filename=r.filename_hr) for i, r in sel.iterrows()]
for q in queue:
    q["modality"] = "both"


# ---- extra positive controls (STE_, LNGQT too rare among unseen fold-10 records) ----
if os.environ.get("EXTRA_POS") == "1":
    _codes = db.scp_codes.apply(ast.literal_eval)
    _done = {q["ecg_id"] for q in queue if q["role"] == "positive"}
    _extra = []
    for kind, cl in {"st_elev": ["STE_"], "qt_long": ["LNGQT"]}.items():
        sel = db[_codes.apply(lambda c: any(k in c for k in cl)) & db.strat_fold.isin([9, 10])
                 & (db.validated_by_human == True) & ~db.index.isin(TRAIN_IDS) & ~db.index.isin(_done)]  # noqa: E712
        sel = sel.sample(min(40, len(sel)), random_state=1)
        print(f"additional positive controls {kind}: {len(sel)} (folds 9-10, never seen)", flush=True)
        _extra += [dict(sample_id=f"pos_{kind}_{i}", ecg_id=int(i), kind=kind, level_idx=-1, role="positive",
                        block="B2", scp=",".join(_codes[i].keys()), filename=r.filename_hr) for i, r in sel.iterrows()]
    queue = [dict(it, modality=m) for m in ["both"] for it in _extra]
RESP = f"{OUT}/responses" + ("_extrapos" if os.environ.get("EXTRA_POS") == "1" else "") + ".jsonl"
seen = {}
if os.path.exists(RESP):
    for line in open(RESP):
        try:
            r = json.loads(line); seen[r["sample_id"]] = r
        except Exception:
            pass
todo = [q for q in queue if q["sample_id"] not in seen]
print(f"queue: {len(queue)} | done: {len(queue) - len(todo)} | remaining: {len(todo)}", flush=True)

# ---------------- inputs ----------------
import wfdb  # noqa: E402
from ecg_render import render  # noqa: E402
_z = {}


def signal_of(it):
    if it["role"] == "positive":
        x, meta = wfdb.rdsamp(os.path.join(PTBXL, it["filename"]))
        names = [s.upper() for s in meta["sig_name"]]
        return x[:, [names.index(l.upper()) for l in PTBXL_ORDER]]
    if it["ecg_id"] not in _z:
        _z.clear(); _z[it["ecg_id"]] = np.load(f"{CF}/signals/{it['ecg_id']}.npz")
    z = _z[it["ecg_id"]]
    return z["original"] if it["kind"] == "original" else z[it["sample_id"]]


def prepare(it):
    x = np.asarray(signal_of(it), float)
    png = f"{OUT}/png/{it['sample_id']}.png"
    if not os.path.exists(png):
        render(x, 500, png)
    sig = x[:5000, TO_MIMIC]          # GEM reads WFDB columns by position (MIMIC-IV-ECG order)
    return sig, png


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


# sanity check: one original, one real AF
for q in queue[:2]:
    txt, dt = ask(*prepare(dict(q)))
    print(f"--- TEST {q['sample_id']} ({dt:.0f}s)\n{txt[:900]}\n", flush=True)
    if len(txt) < 30:
        raise SystemExit("STOP: empty or inconsistent response")

# ---------------- run ----------------
n, t_loop = 0, time.time()
with open(RESP, "a") as f:
    for q in todo:
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
            print("time budget reached: re-run sbatch gem_job.slurm", flush=True); break
        try:
            txt, dt = ask(*prepare(dict(q)))
        except Exception as e:
            print("skipped:", q["sample_id"], str(e)[:200], flush=True); continue
        r = {k: v for k, v in q.items() if k != "filename"}
        r.update(answer=txt, seconds=dt, model="GEM-7B")
        f.write(json.dumps(r) + "\n"); f.flush(); seen[r["sample_id"]] = r; n += 1
        if n % 100 == 0:
            el = time.time() - t_loop
            print(f"{n}/{len(todo)}  {el/60:.0f} min  {el/n:.1f} s/response  (~{(len(todo)-n)*el/n/3600:.1f} h left)", flush=True)
left = [q for q in queue if q["sample_id"] not in seen]
print("DONE" if not left else f"{len(left)} requests left -> re-run sbatch gem_job.slurm", flush=True)
import subprocess  # noqa: E402
subprocess.run(["zip", "-qj", f"{WORK}/main_gem" + ("_extrapos" if os.environ.get("EXTRA_POS") == "1" else "") + "_resultats.zip", RESP])
print("file to send me:", f"{WORK}/main_gem" + ("_extrapos" if os.environ.get("EXTRA_POS") == "1" else "") + "_resultats.zip", flush=True)
