"""Annotation v3: 100 NEW answers (never shown in earlier annotation rounds), English only, no AI."""
import sys, re, copy, numpy as np, pandas as pd, openpyxl
sys.path[:0] = ["coding", "analysis", "ecgcf"]
from faith_parse import decision, evidence_code, decision_t_strict, final_answer
from denial import denies_st_anterior, denies_qt
R = "results/"; A2 = "<previous-annotation-round, not distributed>/"; OUT = "validation/"
rng = np.random.default_rng(2026)
man = pd.read_csv(R + "r1/manifest.csv").set_index("sample_id")
src = {"ECG-R1": ("r1/responses.jsonl", "block=='A' and modality=='signal'"),
       "PULSE": ("pulse/responses.jsonl", "block=='A'"), "GEM": ("gem/responses.jsonl", None)}
used = set()
k2 = pd.read_csv(A2 + "annotation_300_KEY_PRIVE.csv"); used |= set(zip(k2.model, k2.sample_id))
k1 = pd.read_csv(R + "annotation_150_KEY.csv"); used |= set(zip(k1.model, k1.sample_id))
kl = pd.read_csv(R + "annotation_lot1_KEY.csv"); used |= {("PULSE", s) for s in kl.sample_id}
EDITS = {"st_elev": 7, "qt_long": 5, "af": 4, "t_inv": 4, "qrs_wide": 4}   # 24 edits + 5 controls + 4 originals = 33 per model
TARGETS = list(EDITS)
rows = []
for model, (f, q) in src.items():
    D = pd.read_json(R + f, lines=True)
    if q: D = D.query(q)
    D = D[D.answer.astype(str).str.strip().str.len() > 20]
    D = D[[(model, s) not in used for s in D.sample_id]].drop_duplicates("sample_id")
    keep = D.sample_id.map(lambda s: bool(man.keep.get(s, True)) if s in man.index else True)
    D = D[keep]
    for kind, n in EDITS.items():
        pool = D[(D.role == "edit") & (D.kind == kind) & (D.level_idx >= 2)]
        for _, r in pool.sample(n, random_state=int(rng.integers(1e9))).iterrows():
            rows.append(dict(model=model, sample_id=r.sample_id, role="edit", kind=kind, target=kind, answer=r.answer))
    for role, n in [("control", 5), ("original", 4)]:
        pool = D[D.role == role]
        for _, r in pool.sample(n, random_state=int(rng.integers(1e9))).iterrows():
            rows.append(dict(model=model, sample_id=r.sample_id, role=role, kind=r.kind, target=TARGETS[int(rng.integers(5))], answer=r.answer))
T = pd.DataFrame(rows)
# 100 items: add one extra ST edit from a random model
extra = pd.read_json(R + src["GEM"][0], lines=True)
extra = extra[(extra.role == "edit") & (extra.kind == "st_elev") & (extra.level_idx >= 2) & ~extra.sample_id.isin(T.sample_id)]
extra = extra[[("GEM", s) not in used for s in extra.sample_id]].sample(1, random_state=7).iloc[0]
T = pd.concat([T, pd.DataFrame([dict(model="GEM", sample_id=extra.sample_id, role="edit", kind="st_elev", target="st_elev", answer=extra.answer)])])
T = T.sample(frac=1, random_state=11).reset_index(drop=True)
T["code_id"] = [f"C{i+1:03d}" for i in range(len(T))]
assert len(T) == 100 and not any((m, s) in used for m, s in zip(T.model, T.sample_id))

def display(model, a):
    a = str(a)
    if model == "ECG-R1":   # same presentation as v2: reasoning then answer
        a = a.replace("<think>", "RAISONNEMENT :\n").replace("</think>", "\n\nRÉPONSE :\n").replace("<answer>", "").replace("</answer>", "")
    return a.strip()
T["display"] = [display(m, a) for m, a in zip(T.model, T.answer)]
# private key (coder v5)
def codes(a, tg):
    ev = evidence_code(a, tg)
    return dict(Q1=int(decision(a, tg)), Q2=int(ev["evidence"]),
                Q3=(int(ev["grounded"]) if ev["evidence"] else None) if tg in ("st_elev", "t_inv") else None,
                Q4=int(denies_st_anterior(a) if tg == "st_elev" else denies_qt(a)) if tg in ("st_elev", "qt_long") else None,
                Q5=(int(decision_t_strict(a)) if tg == "t_inv" else int(bool(re.search(r"flutter", final_answer(a).lower())))) if tg in ("t_inv", "af") else None)
key = pd.concat([T[["code_id", "model", "sample_id", "role", "kind", "target"]],
                 pd.DataFrame([codes(a, t) for a, t in zip(T.answer, T.target)])], axis=1)
key.to_csv(OUT + "annotation_v3_KEY_PRIVE.csv", index=False)
print(T.groupby(["model", "role"]).size().to_string()); print(T.target.value_counts().to_dict())

LAB = {}
wb0 = openpyxl.load_workbook(A2 + "Annotation_v2_Souaad.xlsx")
for r in range(2, 286):
    v = wb0["Annotation"].cell(r, 2).value
    for t, kw in [("st_elev", "ST"), ("af", "Fibrillation"), ("qt_long", "QT long"), ("qrs_wide", "QRS"), ("t_inv", "onde T")]:
        if kw in v: LAB.setdefault(t, v)
GUIDE = {
 1: "Annotation des réponses des modèles ECG — version 3 (100 réponses nouvelles)",
 5: "1. Travaille SEUL(E). Ne regarde pas les réponses de l'autre annotateur ni aucun codage automatique.",
 6: "2. AUCUN outil d'IA, ni pour décider, NI POUR TRADUIRE, NI POUR REMPLIR le fichier (pas de ChatGPT, Gemini, DeepL, Google Traduction…). Tape toi-même chaque réponse dans Excel.",
 7: "3. Les réponses sont en anglais : utilise uniquement le lexique (onglet « Lexique »). Si un mot manque, écris-le dans « Commentaire » et réponds au mieux.",
 8: "4. Pour chaque ligne, la colonne « Anomalie à rechercher » dit QUELLE anomalie évaluer. Remplis les cellules jaunes (Q1 à Q5) avec 1, 0 ou NA. Compte 1 à 2 min par réponse (2 séances de 50).",
 9: "5. Ne parle pas des réponses avec l'autre annotateur avant que les deux fichiers soient envoyés.",
 16: "Q2 – Preuve précise : la réponse donne-t-elle une preuve observable de CETTE anomalie, au-delà du simple nom ? Une description NORMALE (« QRS < 100 ms », « T waves normal », « QTc 420 ms, normal ») n'est PAS une preuve de l'anomalie = 0.",
 22: "Q4 – Déni explicite (ST et QT seulement, sinon NA) : la réponse dit-elle explicitement que cette composante est NORMALE dans la zone d'intérêt ?",
 23: "      ST : seulement une phrase qui NOMME V1–V4 ou la région antérieure/précordiale et dit que le ST y est normal, isoélectrique, sans sus-décalage, sans changement ou sans déviation (ex. « V2 to V4 … no ST elevation », « normal ST segments in V1–V4 »). Un déni limité à d'autres dérivations (II, III, aVF, aVR, aVL, V5–V6) = 0.   QT : « QT/QTc normal / within normal limits / not prolonged » = 1.",
 26: "      FA : la réponse AFFIRME-t-elle un FLUTTER auriculaire ? 1/0 (« ruling out atrial fibrillation or flutter » = négation = 0)",
}
EXTRA_LEX = [("upright T waves", "ondes T positives (normales)"), ("tented / peaked T waves", "ondes T pointues (hyperkaliémie)"),
             ("flattening", "aplatissement"), ("ST-segment deviation", "déviation (décalage) du segment ST"),
             ("declination / downsloping ST", "ST descendant"), ("upslope / upsloping ST", "ST ascendant"),
             ("horizontal ST", "ST horizontal"), ("early repolarization", "repolarisation précoce"),
             ("narrow QRS / QRS duration < 100 ms", "QRS fin / durée du QRS < 100 ms (normal)"),
             ("first-degree AV block / prolonged PR", "BAV du 1er degré / PR allongé"),
             ("premature atrial complexes", "extrasystoles auriculaires"), ("left ventricular hypertrophy (LVH)", "hypertrophie ventriculaire gauche (HVG)"),
             ("poor / abnormal R-wave progression", "mauvaise progression de l'onde R"), ("secondary repolarization abnormality", "trouble secondaire de la repolarisation"),
             ("does not suggest / not indicative of", "ne suggère pas / n'évoque pas (= négation)"), ("consistent with / suggestive of", "compatible avec / évocateur de (= affirmé)"),
             ("mimic", "imiter (« can mimic ischemia » = peut ressembler à une ischémie, sans l'affirmer)"),
             ("prolonged", "allongé"), ("within normal range", "dans les limites normales"), ("lead", "dérivation")]
for who in ["Souaad", "Mostafa"]:
    wb = openpyxl.load_workbook(A2 + f"Annotation_v2_{who}.xlsx")
    g = wb["Consignes"]
    for r, txt in GUIDE.items(): g.cell(r, 1).value = txt
    g.cell(2, 1).value = f"Annotateur / annotatrice : {who}"
    lx = wb["Lexique"]
    for e, f in EXTRA_LEX: lx.append([e, f])
    ws = wb["Annotation"]
    style = [copy.copy(ws.cell(2, c)._style) for c in range(1, 10)]
    h = ws.row_dimensions[2].height
    ws.delete_rows(2, ws.max_row)
    for i, r in T.iterrows():
        rr = i + 2
        for c, v in enumerate([r.code_id, LAB[r.target], r.display, None, None, None, None, None, None], 1):
            ws.cell(rr, c).value = v; ws.cell(rr, c)._style = copy.copy(style[c - 1])
        ws.row_dimensions[rr].height = h
    ws.data_validations.dataValidation[0].sqref = openpyxl.worksheet.cell_range.MultiCellRange("D2:H101")
    ws.cell(1, 11).value = "Progression"; ws.cell(2, 11).value = '=COUNTA(D2:D101)&" / 100 réponses codées (Q1)"'
    wb.properties.creator = "ECG-CF"; wb.properties.lastModifiedBy = ""
    wb.save(OUT + f"Annotation_v3_{who}.xlsx")
print("ok")
