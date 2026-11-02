"""ECG-CF review experiments, ECG-R1-8B (signal), Jean Zay.
Georgia external replication + PTB-XL reciprocal-ST edit. Resumes from OUT/responses.jsonl.
Needs build_ext.py to have been run (ext/cf_georgia, ext/cf_recip, ext/pos_georgia)."""
import json, os, sys, time, subprocess
import numpy as np, pandas as pd
T_START = time.time()
BATCH = int(os.environ.get("ECGCF_BATCH", 16))
MAX_NEW_TOKENS = 2048
DTYPE = os.environ.get("ECGCF_DTYPE", "bfloat16")
TIME_BUDGET_H = float(os.environ.get("ECGCF_HOURS", 7.5))
MOCK = os.environ.get("PILOT_MOCK") == "1"
WORK = os.environ["ECGCF_WORK"]; R1_DIR = os.environ.get("ECGR1_DIR", ""); GDIR = os.environ["GEORGIA_DIR"]
PTBXL_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
MIMIC_ORDER = ["I","II","III","aVR","aVF","aVL","V1","V2","V3","V4","V5","V6"]
TO_MIMIC = [PTBXL_ORDER.index(l) for l in MIMIC_ORDER]
OUT = f"{WORK}/ext/r1"
os.makedirs(f"{OUT}/wfdb", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_queue import build_queue, signal_of
queue = [dict(q, modality="signal") for q in build_queue(WORK, GDIR, full=True)]
RESP = f"{OUT}/responses.jsonl"
KEY = lambda r: r["sample_id"]
seen = {}
if os.path.exists(RESP):
    for line in open(RESP):
        try:
            r = json.loads(line); seen[KEY(r)] = r
        except Exception:
            pass
todo = [q for q in queue if KEY(q) not in seen]
print(f"queue: {len(queue)} | done: {len(queue)-len(todo)} | remaining: {len(todo)}", flush=True)
print(pd.DataFrame(queue).groupby(["block", "role"]).size().to_string(), flush=True)

import wfdb
def to_wfdb(sig, name):
    path = f"{OUT}/wfdb/{name}"
    if not os.path.exists(path + ".hea"):
        x = np.asarray(sig, float)[:5000, TO_MIMIC]
        wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=MIMIC_ORDER, p_signal=x, fmt=["16"] * 12,
                    adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=f"{OUT}/wfdb")
    return path
def prepare(it):
    it["ecg"] = to_wfdb(signal_of(it, WORK, GDIR), it["sample_id"])
    return it
t0 = prepare(dict(queue[0]))
err = np.abs(wfdb.rdsamp(t0["ecg"])[0] - signal_of(t0, WORK, GDIR)[:5000, TO_MIMIC]).max()
print(f"WFDB round-trip check: {err*1000:.2f} uV"); assert err < 0.001

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
probe = [prepare(dict(q)) for q in [queue[0], next(q for q in queue if q["role"] == "positive")]]
out, dt = run_batch(probe)
for it, (txt, fin) in zip(probe, out):
    print(f"--- {it['sample_id']} ({fin})\n{txt[-600:]}\n", flush=True)
if not all(len(t) > 40 for t, _ in out):
    raise SystemExit("STOP: inconsistent output")

n, t_loop = 0, time.time()
with open(RESP, "a") as f:
    for i in range(0, len(todo), BATCH):
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
            print("time budget reached: re-run sbatch ext_r1.slurm"); break
        chunk = [prepare(dict(q)) for q in todo[i:i + BATCH]]
        try:
            out, dt = run_batch(chunk)
        except Exception as e:
            print("batch skipped:", str(e)[:200], flush=True); continue
        for it, (txt, fin) in zip(chunk, out):
            r = {k: v for k, v in it.items() if k != "ecg"}
            r.update(answer=txt, finish=fin, seconds=dt / len(chunk), model="ECG-R1")
            f.write(json.dumps(r) + "\n"); seen[KEY(r)] = r
        f.flush(); n += len(chunk)
        if (i // BATCH) % 10 == 0:
            el = time.time() - t_loop
            print(f"{n}/{len(todo)}  {el/60:.0f} min  {el/max(n,1):.1f} s/response  (~{(len(todo)-n)*el/max(n,1)/3600:.1f} h left)", flush=True)
left = [q for q in queue if KEY(q) not in seen]
print("DONE" if not left else f"{len(left)} requests left -> re-run sbatch ext_r1.slurm", flush=True)
Z = f"{WORK}/ext_r1_resultats.zip"
subprocess.run(["zip", "-qj", Z, RESP, f"{WORK}/ext/cf_georgia/manifest.csv", f"{WORK}/ext/cf_georgia/base_ecgs.csv",
                f"{GDIR}/selection.json"])
subprocess.run(["bash", "-c", f"cp {WORK}/ext/cf_recip/manifest.csv {OUT}/recip_manifest.csv && zip -qj {Z} {OUT}/recip_manifest.csv"])
print("file to send me:", Z, flush=True)
