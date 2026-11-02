"""ECG-CF main experiment, ECG-R1-8B, Slurm version (HPC cluster; generated from ecg_cf_main_ecgr1.ipynb).
Run from the inference/ folder via job.slurm. Resumes automatically from OUT/responses.jsonl."""
import subprocess, os, sys


BATCH = int(os.environ.get("ECGCF_BATCH", 16))                    # 2 x T4: 8 requests at a time (pilot: 4). If "out of memory": 4
MAX_NEW_TOKENS = 2048        # authors' value; the pilot never reached 1,536
DTYPE = os.environ.get("ECGCF_DTYPE", "bfloat16")   # A100/H100: bfloat16 like the authors
TIME_BUDGET_H = float(os.environ.get("ECGCF_HOURS", 19.0))         # clean stop before the Kaggle 12 h limit
N_POS_PER_CLASS = 20
N_SUBSET_IMAGE = 100
SUPRA = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3}   # levels above the threshold
CONTROLS = [("ctrl_amp", 2), ("ctrl_noise", 2), ("ctrl_wander", 2)]
POS_CODES = {"st_elev": ["STE_"], "t_inv": ["INVT"], "qt_long": ["LNGQT"],
             "qrs_wide": ["CLBBB", "CRBBB"], "af": ["AFIB"]}
PTBXL_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
MIMIC_ORDER = ["I","II","III","aVR","aVF","aVL","V1","V2","V3","V4","V5","V6"]
TO_MIMIC = [PTBXL_ORDER.index(l) for l in MIMIC_ORDER]
import os, sys, json, time, glob, shutil, ast
import numpy as np, pandas as pd
T_START = time.time()
MOCK = os.environ.get("PILOT_MOCK") == "1"
WORK = os.environ["ECGCF_WORK"]; R1_DIR = os.environ["ECGR1_DIR"]
OUT = f"{WORK}/main_r1"
for d in ["wfdb", "png"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
TRAIN_IDS = set(json.load(open("ptbxl_ids_in_training.json")))
if MOCK:
    TRAIN_IDS = {1, 2}                     # test: dummy data
print("PTB-XL ECGs seen during training:", len(TRAIN_IDS))

def find_dir_with(name, root="/kaggle/input", max_depth=8):
    for dirpath, dirnames, files in os.walk(root, followlinks=True):
        if name in files:
            return dirpath
        if dirpath.count(os.sep) - root.count(os.sep) > max_depth:
            dirnames[:] = []
PTBXL = os.environ.get("PTBXL_DIR") or find_dir_with("ptbxl_database.csv")
assert PTBXL, "PTB-XL not found: add it via Add Input"
db = pd.read_csv(os.path.join(PTBXL, "ptbxl_database.csv"), index_col="ecg_id")
RESP = f"{OUT}/responses" + ("_extrapos" if os.environ.get("EXTRA_POS") == "1" else "") + ".jsonl"
KEY = lambda r: (r["sample_id"], r["modality"])
prev = [p for p in glob.glob("/kaggle/input/**/main_r1/responses.jsonl", recursive=True)]
seen = {}
for p in prev + ([RESP] if os.path.exists(RESP) else []):
    for line in open(p):
        try:
            r = json.loads(line); seen[KEY(r)] = r
        except Exception:
            pass                                   # line truncated by an abrupt stop
with open(RESP, "w") as f:
    for r in seen.values():
        f.write(json.dumps(r) + "\n")
print(f"responses recovered: {len(seen)} (from {len(prev)} previous run(s))")
prev_cf = glob.glob("/kaggle/input/**/main_r1/cf/manifest.csv", recursive=True)
if prev_cf and not os.path.exists(f"{OUT}/cf/manifest.csv"):
    shutil.copytree(os.path.dirname(prev_cf[0]), f"{OUT}/cf", dirs_exist_ok=True)
    print("edited ECGs recovered from the previous run")

CF = f"{OUT}/cf"
if not os.path.exists(f"{CF}/manifest.csv"):
    subprocess.run([sys.executable, "run_ptbxl.py", "--ptbxl", PTBXL, "--n", "100000", "--out", CF], check=True)
man = pd.read_csv(f"{CF}/manifest.csv")
base = pd.read_csv(f"{CF}/base_ecgs.csv")
inc = base[base.excluded.fillna("") == ""].ecg_id.astype(int)
unseen = [e for e in inc if e not in TRAIN_IDS]
exposed = [e for e in inc if e in TRAIN_IDS]
order = np.random.default_rng(2026).permutation(sorted(unseen)).tolist()   # fixed, random order
subset = set(order[:N_SUBSET_IMAGE])
print(f"ECGs included: {len(inc)} | never seen (block A): {len(unseen)} | already seen (block D): {len(exposed)}")
tab = man[man.ecg_id.isin(unseen)].groupby(["kind", "level_idx"]).agg(kept=("keep", "mean"), crit=("criterion_met", "mean"))
print(tab.round(2).unstack("level_idx").to_string())

import wfdb
from ecg_render import render
def to_wfdb(sig, name):
    path = f"{OUT}/wfdb/{name}"
    if not os.path.exists(path + ".hea"):
        x = np.asarray(sig, float)[:5000, TO_MIMIC]
        wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=MIMIC_ORDER, p_signal=x, fmt=["16"] * 12,
                    adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=f"{OUT}/wfdb")
    return path
def to_png(sig, name):
    path = f"{OUT}/png/{name}.png"
    if not os.path.exists(path):
        render(np.asarray(sig, float), 500, path)
    return path
KEEP = man.set_index("sample_id").keep.to_dict()
def variants(ecg_id, full):
    v = [("original", 0)]
    if full:
        v += [(k, li) for k, spec in __import__("ecg_cf").CATALOGUE.items() if not k.startswith("ctrl")
              for li in range(1, len(spec["levels"]) + 1)]
    else:
        v += list(SUPRA.items())
    v += CONTROLS
    out = []
    for k, li in v:
        sid = f"{ecg_id}_original" if k == "original" else f"{ecg_id}_{k}_L{li}"
        if k != "original" and not KEEP.get(sid, False):
            continue
        role = "original" if k == "original" else ("control" if k.startswith("ctrl") else "edit")
        out.append(dict(sample_id=sid, ecg_id=int(ecg_id), kind=k, level_idx=li, role=role))
    return out
queue = []
for e in order:                                                   # A
    queue += [dict(it, block="A", modality="signal") for it in variants(e, True)]
codes = db.scp_codes.apply(ast.literal_eval)
pos = []
for kind, cl in POS_CODES.items():                                # B
    sel = db[codes.apply(lambda c: any(k in c for k in cl)) & (db.strat_fold == 10)
             & (db.validated_by_human == True) & ~db.index.isin(TRAIN_IDS)]
    sel = sel.sample(min(N_POS_PER_CLASS, len(sel)), random_state=0)
    pos += [dict(sample_id=f"pos_{kind}_{i}", ecg_id=int(i), kind=kind, level_idx=-1, role="positive",
                 scp=",".join(codes[i].keys()), filename=r.filename_hr) for i, r in sel.iterrows()]
    print(f"positive controls {kind:9s}: {len(sel)} ({'/'.join(cl)}, never seen)")
for m in ["signal", "image", "both"]:
    queue += [dict(it, block="B", modality=m) for it in pos]
for e in order[:N_SUBSET_IMAGE]:                                  # C
    for m in ["image", "both"]:
        queue += [dict(it, block="C", modality=m) for it in variants(e, False)]
for e in sorted(exposed):                                         # D
    queue += [dict(it, block="D", modality="signal") for it in variants(e, False)]

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
    queue = [dict(it, modality=m) for m in ["signal", "image", "both"] for it in _extra]
todo = [q for q in queue if KEY(q) not in seen]
print(f"\ntotal queue: {len(queue)} requests | done: {len(queue) - len(todo)} | remaining: {len(todo)}")
print(pd.DataFrame(queue).groupby(["block", "modality"]).size().to_string())

_sig_cache = {}
def signal_of(it):
    if it["role"] == "positive":
        x, meta = wfdb.rdsamp(os.path.join(PTBXL, it["filename"]))
        names = [s.upper() for s in meta["sig_name"]]
        return x[:, [names.index(l.upper()) for l in PTBXL_ORDER]]
    if it["ecg_id"] not in _sig_cache:
        _sig_cache.clear(); _sig_cache[it["ecg_id"]] = np.load(f"{CF}/signals/{it['ecg_id']}.npz")
    z = _sig_cache[it["ecg_id"]]
    return z["original"] if it["kind"] == "original" else z[it["sample_id"]]
def prepare(it):
    x = signal_of(it)
    if it["modality"] in ("signal", "both"):
        it["ecg"] = to_wfdb(x, it["sample_id"])
    if it["modality"] in ("image", "both"):
        it["img"] = to_png(x, it["sample_id"])
    return it
# WFDB round-trip check on the first item
t0 = prepare(dict(next(q for q in queue if q["modality"] == "signal")))
err = np.abs(wfdb.rdsamp(t0["ecg"])[0] - signal_of(t0)[:5000, TO_MIMIC]).max()
print(f"\nWFDB round-trip check: {err*1000:.2f} uV"); assert err < 0.001

DEFAULT = "Interpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence."
PROMPTS = {"signal": "<ecg> \n" + DEFAULT, "image": "<image>\n" + DEFAULT, "both": "<image>\n<ecg> \n" + DEFAULT}
if not MOCK:
    from huggingface_hub import hf_hub_download
    for m, f in [("signal", "image_missing100"), ("image", "ecg_missing100"), ("both", "full")]:
        try:
            p = hf_hub_download("PKUDigitalHealth/ECG-Protocol-Guided-Grounding-CoT",
                                f"ecg_jsons/test_set/ecg-grounding-test-mimiciv_{f}.jsonl", repo_type="dataset")
            row = json.loads(open(p).readline())
            msgs = row.get("messages") or row.get("conversations")
            u = [x for x in msgs if x.get("role", x.get("from")) in ("user", "human")][0]
            PROMPTS[m] = u.get("content", u.get("value"))
        except Exception as e:
            print(f"{m}: official prompt not read ({e}) -> default prompt")
for m, p in PROMPTS.items():
    print(f"{m:6s}: {p!r}")

if not MOCK:
    import torch
    sys.path.insert(0, R1_DIR)
    # 7a. encoder structure: file at the path expected by the model config
    ck_rel = "ecg_coca/checkpoint/cpt_wfep_epoch_20.pt"
    ck = os.path.join(R1_DIR, ck_rel)
    if not os.path.exists(ck):
        from ecg_coca.open_clip.model import _build_ecg_tower
        mcfg = json.load(open(os.path.join(R1_DIR, "ecg_coca/open_clip/model_configs/coca_ViT-B-32.json")))
        tower = _build_ecg_tower(embed_dim=mcfg["embed_dim"], ecg_cfg=mcfg["ecg_cfg"])
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({"state_dict": {"module.ecg." + k: v for k, v in tower.state_dict().items()}}, ck)
        PLACEHOLDER = {k: v.clone() for k, v in tower.state_dict().items()}
    os.environ.update(ECG_TOWER_PATH=ck, ECG_SEQ_LENGTH="5000", ECG_PATCH_SIZE="50",
                      ECG_PROJECTOR_TYPE="mlp2x_gelu", ECG_MODEL_CONFIG="coca_ViT-B-32",
                      FREEZE_ECG_TOWER="True", FREEZE_ECG_PROJECTOR="True", IMAGE_MAX_TOKEN_NUM="768")
    os.chdir(R1_DIR)                     # the config references the encoder's relative path
    import ecg_r1.register               # registers the model and the "ecg_r1" template
    from swift.llm import PtEngine, InferRequest, RequestConfig
    engine = PtEngine("PKUDigitalHealth/ECG-R1-8B-RL", torch_dtype=getattr(torch, DTYPE),
                      model_type="ecg_r1", use_hf=True, device_map="auto", max_batch_size=BATCH)
    os.chdir(WORK)
    print("model loaded; dtype", DTYPE, "; GPU:",
          [round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())], "GB")

if not MOCK:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx = json.load(open(hf_hub_download("PKUDigitalHealth/ECG-R1-8B-RL", "model.safetensors.index.json")))
    params = dict(engine.model.named_parameters())
    keys = [k for k in idx["weight_map"] if k.startswith(("model.ecg_tower", "model.ecg_projector"))]
    tower_params = [k for k in params if ".ecg_tower." in k or ".ecg_projector." in k]
    print(f"published encoder+projector keys: {len(keys)} | parameters in the loaded model: {len(tower_params)}")
    ok, bad = 0, []
    for k in keys[:: max(1, len(keys) // 12)]:
        f = hf_hub_download("PKUDigitalHealth/ECG-R1-8B-RL", idx["weight_map"][k])
        with safe_open(f, "pt") as sf:
            ref = sf.get_tensor(k).float()
        cand = [p for n, p in params.items() if n.endswith(k.split("model.", 1)[1])]
        if cand and torch.allclose(cand[0].detach().float().cpu(), ref, atol=1e-2, rtol=1e-2):
            ok += 1
        else:
            bad.append(k)
    print(f"weights identical to published weights: {ok}/{ok + len(bad)}", "" if not bad else f"-> different: {bad[:3]}")
    assert not bad, "STOP: the signal encoder does not have the published weights"

def run_batch(batch):
    if MOCK:
        pool = ["<think>Sinus rhythm. The ST segments are isoelectric.</think><answer>Sinus rhythm</answer>",
                "<think>Irregularly irregular rhythm with absence of P waves.</think><answer>Atrial fibrillation</answer>",
                "<think>ST elevation in V1-V4.</think><answer>Sinus rhythm; Anterior ST elevation, consider myocardial injury</answer>",
                "<think>The QT interval is prolonged with a QTc of 520 ms.</think><answer>Sinus rhythm; Prolonged QT interval</answer>"]
        return [(pool[hash(b["sample_id"] + b["modality"]) % len(pool)], "stop") for b in batch], 0.0
    reqs = []
    for it in batch:
        kw = {}
        if it["modality"] in ("signal", "both"):
            kw["objects"] = {"ecg": [it["ecg"]]}
        if it["modality"] in ("image", "both"):
            kw["images"] = [it["img"]]
        reqs.append(InferRequest(messages=[{"role": "user", "content": PROMPTS[it["modality"]]}], **kw))
    t0 = time.time()
    res = engine.infer(reqs, RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=0), use_tqdm=False)
    return [(r.choices[0].message.content, r.choices[0].finish_reason) for r in res], time.time() - t0
probe = [prepare(dict(next(q for q in queue if q["modality"] == m))) for m in ["signal", "image", "both"]]
out, dt = run_batch(probe)
for it, (txt, fin) in zip(probe, out):
    print(f"--- {it['sample_id']} [{it['modality']}] ({fin})\n{txt[-700:]}\n")
if not all(len(t) > 40 for t, _ in out):
    raise SystemExit("STOP: inconsistent output. Set DTYPE = 'bfloat16' or BATCH = 4 and re-run.")

n, t_loop = 0, time.time()
with open(RESP, "a") as f:
    for i in range(0, len(todo), BATCH):
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
            print("time budget reached: re-run the notebook (see header) to continue"); break
        chunk = [prepare(dict(q)) for q in todo[i:i + BATCH]]
        try:
            out, dt = run_batch(chunk)
        except Exception as e:                  # e.g. memory: skip this batch, it will be retried
            print("batch skipped:", str(e)[:200]); continue
        for it, (txt, fin) in zip(chunk, out):
            r = {k: v for k, v in it.items() if k not in ("filename",)}
            r.update(answer=txt, finish=fin, seconds=dt / len(chunk))
            f.write(json.dumps(r) + "\n"); seen[KEY(r)] = r
        f.flush(); n += len(chunk)
        if (i // BATCH) % 10 == 0:
            el = time.time() - t_loop
            print(f"{n}/{len(todo)}  {el/60:.0f} min  {el/max(n,1):.1f} s/response  "
                  f"(~{(len(todo)-n)*el/max(n,1)/3600:.1f} h left)")
left = [q for q in queue if KEY(q) not in seen]
print("DONE" if not left else f"{len(left)} requests left -> re-run the notebook")

from faith_parse import decision, evidence_code
R = pd.DataFrame(list(seen.values()))
A = R[(R.block == "A") & (R.modality == "signal")]
O = A[A.role == "original"].set_index("ecg_id").answer
rows = []
for _, r in A[A.role == "edit"].iterrows():
    if r.ecg_id not in O.index: continue
    I = decision(r.answer, r.kind) and not decision(O[r.ecg_id], r.kind)
    ev = evidence_code(r.answer, r.kind)
    crit = man.set_index("sample_id").criterion_met.get(r.sample_id, np.nan)
    rows.append(dict(kind=r.kind, level=r.level_idx, crit=crit, I=I, **ev))
E = pd.DataFrame(rows)
if len(E):
    print("DIR by edit and level:")
    print(E.pivot_table(index="kind", columns="level", values="I", aggfunc="mean").round(2).to_string())
    print("\nTA (DIR criterion met - not met):")
    print(E.groupby("kind").apply(lambda g: g[g.crit == True].I.mean() - g[g.crit == False].I.mean()).round(2).to_string())
    f = E[E.I]
    print("\ngiven that the decision changed: evidence / localized / mislocalized")
    print(f.groupby("kind")[["evidence", "grounded", "misloc"]].mean().round(2).to_string())
ctl = A[A.role == "control"]
if len(ctl):
    chg = [any(decision(r.answer, k) and not decision(O.get(r.ecg_id, ""), k) for k in ["st_elev","t_inv","qt_long","qrs_wide","af"])
           for _, r in ctl.iterrows()]
    print(f"\ncontrols with a new abnormality: {np.mean(chg):.1%} (n={len(ctl)})")
print(f"\ntruncated: {(R.finish == 'length').mean():.1%} | total responses: {len(R)}")

import shutil
subprocess.run(["zip", "-qj", f"{WORK}/main_r1" + ("_extrapos" if os.environ.get("EXTRA_POS") == "1" else "") + "_resultats.zip", RESP, f"{CF}/manifest.csv", f"{CF}/base_ecgs.csv"])
print("file to send me:", f"{WORK}/main_r1" + ("_extrapos" if os.environ.get("EXTRA_POS") == "1" else "") + "_resultats.zip")