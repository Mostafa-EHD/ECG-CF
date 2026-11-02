# Guide d'annotation (validation du codeur, version 3)

Texte fourni aux annotateurs (français), tel quel.

> Note: the guide asked each annotator to work alone. In practice the two authors coded the 100
> answers together and resolved the 29 codes on which their provisional codes differed by
> discussion; the paper therefore reports agreement between the coder and this consensus, not
> inter-annotator agreement.

Annotation des réponses des modèles ECG — version 3 (100 réponses nouvelles)
Règles générales
1. Travaille SEUL(E). Ne regarde pas les réponses de l'autre annotateur ni aucun codage automatique.
2. AUCUN outil d'IA, ni pour décider, NI POUR TRADUIRE, NI POUR REMPLIR le fichier (pas de ChatGPT, Gemini, DeepL, Google Traduction…). Tape toi-même chaque réponse dans Excel.
3. Les réponses sont en anglais : utilise uniquement le lexique (onglet « Lexique »). Si un mot manque, écris-le dans « Commentaire » et réponds au mieux.
4. Pour chaque ligne, la colonne « Anomalie à rechercher » dit QUELLE anomalie évaluer. Remplis les cellules jaunes (Q1 à Q5) avec 1, 0 ou NA. Compte 1 à 2 min par réponse (2 séances de 50).
5. Ne parle pas des réponses avec l'autre annotateur avant que les deux fichiers soient envoyés.
Les questions
Q1 – Décision : la réponse AFFIRME-t-elle cette anomalie (même avec « possible », « probable », « non spécifique », « cannot be excluded ») ?
      Pour l'onde T : une « anomalie non spécifique de l'onde T », « T wave changes », « repolarization abnormality » ou une « ischémie » comptent comme OUI.
      Pour la FA : seule « atrial fibrillation » compte. Un « flutter auriculaire » = 0 en Q1 (mais 1 en Q5).
      Une négation (« no ST elevation », « does not suggest ischemia », « ruling out ») = NON.
Q2 – Preuve précise : la réponse donne-t-elle une preuve observable de CETTE anomalie, au-delà du simple nom ? Une description NORMALE (« QRS < 100 ms », « T waves normal », « QTc 420 ms, normal ») n'est PAS une preuve de l'anomalie = 0.
      ST et T : l'anomalie du ST ou de l'onde T est décrite avec des dérivations ou une région (ex. « ST elevation in V1–V4 »).
      QT : une valeur chiffrée du QT/QTc est donnée.   QRS : la largeur du QRS est décrite (« wide », « >120 ms »), pas seulement « bloc de branche ».
      FA : absence d'ondes P, rythme irrégulier (RR irréguliers) ou ondes f.   « Infarctus », « ischémie » seuls = NON (c'est un nom, pas une preuve).
Q3 – Localisation (ST et T seulement, sinon NA ; NA aussi si Q2 = 0) : la preuve est-elle placée dans la zone d'intérêt ?
      ST : V1, V2, V3 ou V4 ou « antérieur/antéroseptal/précordial ».   T : V2 à V5 ou « antérieur/latéral/précordial ». Ailleurs seulement = 0.
Q4 – Déni explicite (ST et QT seulement, sinon NA) : la réponse dit-elle explicitement que cette composante est NORMALE dans la zone d'intérêt ?
      ST : seulement une phrase qui NOMME V1–V4 ou la région antérieure/précordiale et dit que le ST y est normal, isoélectrique, sans sus-décalage, sans changement ou sans déviation (ex. « V2 to V4 … no ST elevation », « normal ST segments in V1–V4 »). Un déni limité à d'autres dérivations (II, III, aVF, aVR, aVL, V5–V6) = 0.   QT : « QT/QTc normal / within normal limits / not prolonged » = 1.
Q5 – Nom précis (T et FA seulement, sinon NA) :
      T : la réponse dit-elle explicitement que l'onde T est INVERSÉE ou NÉGATIVE (« T wave inversion », « inverted T waves ») ? 1/0
      FA : la réponse AFFIRME-t-elle un FLUTTER auriculaire ? 1/0 (« ruling out atrial fibrillation or flutter » = négation = 0)
Exemple (ligne fictive, pour le format)
Anomalie : Sus-décalage ST | Texte : « …ST elevation in leads II, III and aVF suggesting inferior injury… »  →  Q1 = 1 ; Q2 = 1 ; Q3 = 0 (inférieur, pas V1–V4) ; Q4 = 0 ; Q5 = NA
Quand tu as fini : enregistre le fichier sous ton nom et envoie-le. Ne modifie pas les colonnes A à C.

## Lexique anglais–français

| Anglais | Français |
|---|---|
| ST elevation / ST-segment elevation | sus-décalage du segment ST |
| ST depression | sous-décalage du ST |
| isoelectric | isoélectrique (sur la ligne de base, normal) |
| T wave inversion / inverted T waves | inversion de l'onde T / ondes T inversées |
| nonspecific T wave abnormality / ST-T changes | anomalie non spécifique de l'onde T / modifications ST-T |
| repolarization abnormality | trouble de la repolarisation |
| ischemia / ischemic changes | ischémie / modifications ischémiques |
| infarction / injury | infarctus / lésion |
| prolonged QT / QTc | QT / QTc allongé |
| within normal limits | dans les limites de la normale |
| wide QRS / QRS widening | QRS large / élargissement du QRS |
| bundle branch block (LBBB/RBBB) | bloc de branche (gauche/droit) |
| intraventricular conduction delay/defect | trouble de conduction intraventriculaire |
| atrial fibrillation | fibrillation auriculaire |
| atrial flutter | flutter auriculaire |
| absent P waves | absence d'ondes P |
| irregular R-R intervals | intervalles RR irréguliers |
| fibrillatory (f) waves | ondes f de fibrillation |
| anterior / inferior / lateral leads | dérivations antérieures / inférieures / latérales |
| cannot be excluded / possible / probable | ne peut être exclu / possible / probable (= affirmé, avec doute) |
| ruling out / no evidence of | éliminant / pas de signe de (= négation) |
| upright T waves | ondes T positives (normales) |
| tented / peaked T waves | ondes T pointues (hyperkaliémie) |
| flattening | aplatissement |
| ST-segment deviation | déviation (décalage) du segment ST |
| declination / downsloping ST | ST descendant |
| upslope / upsloping ST | ST ascendant |
| horizontal ST | ST horizontal |
| early repolarization | repolarisation précoce |
| narrow QRS / QRS duration < 100 ms | QRS fin / durée du QRS < 100 ms (normal) |
| first-degree AV block / prolonged PR | BAV du 1er degré / PR allongé |
| premature atrial complexes | extrasystoles auriculaires |
| left ventricular hypertrophy (LVH) | hypertrophie ventriculaire gauche (HVG) |
| poor / abnormal R-wave progression | mauvaise progression de l'onde R |
| secondary repolarization abnormality | trouble secondaire de la repolarisation |
| does not suggest / not indicative of | ne suggère pas / n'évoque pas (= négation) |
| consistent with / suggestive of | compatible avec / évocateur de (= affirmé) |
| mimic | imiter (« can mimic ischemia » = peut ressembler à une ischémie, sans l'affirmer) |
| prolonged | allongé |
| within normal range | dans les limites normales |
| lead | dérivation |
