"""Download only what ECG-CF needs from the Georgia database (G12EC).

1. all 10,344 headers (.hea, ~1 kB each);
2. selection, deterministic (seed 2026):
   - normal candidates: '#Dx' is sinus rhythm only, adult (>= 18 y), sex known;
   - positive controls per finding (STE, long QT, T inversion, AF, bundle-branch block);
   - the 2,026 G12EC recordings of the ECG-Bench test set are excluded;
3. the .mat signals of the selected records only (~550 files, ~70 MB).

Usage:  python fetch_georgia.py --out $SCRATCH/georgia --exclude georgia_ids_to_exclude.json
Resumable: files already present are not downloaded again.
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from georgia import BASE_URL, N_POS, N_RECORDS, POS_SNOMED, SR, parse_header, rec_name, subfolder

N_NORMAL_CANDIDATES = 320            # ~150 needed after our own measurement checks (HR window etc.)


def get(url, dest, tries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:                    # noqa: BLE001
            if "404" in str(e):
                return False
            time.sleep(2 * (k + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="georgia_ids_to_exclude.json")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = [rec_name(i) for i in range(1, N_RECORDS + 1)]
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        ok = list(ex.map(lambda r: get(f"{BASE_URL}/{subfolder(r)}/{r}.hea", os.path.join(a.out, r + ".hea")), recs))
    print(f"headers: {sum(ok)}/{len(recs)} ({time.time() - t0:.0f} s)", flush=True)
    if sum(ok) < 0.95 * len(recs):
        raise SystemExit("STOP: too many headers missing (connection?). Re-run the same command.")

    excl = set(json.load(open(a.exclude))["g12_bench_test"])
    meta = {}
    for r in recs:
        p = os.path.join(a.out, r + ".hea")
        if os.path.exists(p) and r not in excl:
            meta[r] = parse_header(open(p).read())
    print(f"usable recordings (excluding ECG-Bench test): {len(meta)}")

    rng = np.random.default_rng(2026)
    normals = [r for r, m in meta.items() if m["dx"] == [SR] and m["age"] is not None and m["age"] >= 18
               and m["sex"] in ("M", "F") and abs(m["fs"] - 500) < 1 and m["n"] >= 5000]
    normals = list(rng.permutation(sorted(normals)))
    print(f"normal candidates (sinus rhythm only, adults): {len(normals)}")
    sel = {"normal": normals[:N_NORMAL_CANDIDATES], "positive": {}}
    used = set()
    for kind, codes in POS_SNOMED.items():
        c = sorted(r for r, m in meta.items() if any(x in m["dx"] for x in codes) and r not in used
                   and abs(m["fs"] - 500) < 1 and m["n"] >= 5000)
        pick = list(rng.permutation(c)[:N_POS[kind]])
        used |= set(pick)
        sel["positive"][kind] = [dict(rec=r, dx=meta[r]["dx"]) for r in pick]
        print(f"positive controls {kind:9s}: {len(pick)} (out of {len(c)} available)")
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)

    need = sel["normal"] + [p["rec"] for v in sel["positive"].values() for p in v]
    with ThreadPoolExecutor(a.threads) as ex:
        ok = list(ex.map(lambda r: get(f"{BASE_URL}/{subfolder(r)}/{r}.mat", os.path.join(a.out, r + ".mat")), need))
    print(f"signals downloaded: {sum(ok)}/{len(need)}")
    if not all(ok):
        raise SystemExit("STOP: signals missing. Re-run the same command (automatic resume).")
    print("DONE: selection in", os.path.join(a.out, "selection.json"))


if __name__ == "__main__":
    main()
