# Synth diagnosis (seed 1)

doc_id: `doc_b281c9bd70bedd34`
v1 run: `run_20260924_171941_kaalkram_v1_9ccdc3`
v2 run: `run_20260924_172410_kaalkram_v2_709742`

## 1. Errors

### v1
- `summary.errors` = **0**
- items with `error` set: **0 / 368**
  - (no per-item errors — abstentions are successful `cannot_determine` answers)

### v2
- `summary.errors` = **0**
- items with `error` set: **0 / 368**
  - (no per-item errors — abstentions are successful `cannot_determine` answers)

## 2. Relation / detail counts

### v1 `pred_label` (n=368)
- `cannot_determine`: 367
- `null`: 1

### v2 `pred_label` (n=368)
- `cannot_determine`: 341
- `null`: 27

## 3. v2 graph / build

- events: **14**
- mentions_by_mode: `{'occurs': 13, 'predicted': 7}`
- occurs non-telling: **6**
- frames_by_type: `{'main': 1, 'prediction': 7}`
- edges: **15** by_source=`{'explicit': 4, 'succession': 13, 'lifecycle': 1}`
- removed: **1**
- prompt_version: `v2.1`
- build_events: `{'coref_hard_block': 11, 'coref_cosine_threshold': 2, 'entity_cosine_threshold': 2, 'frame_warning': 1}`
- **FLAG: occurs mentions rare**

- raw LLM mentions in extract cache (sum s0): **13**

## 4. Offsets + build-time gold recall

- mention quote match rate: **20/20 = 1.000**
- gold evidence spans overlapping a mention (>=50%): **12/308 = 0.039**

## 5. Grounding (30 gold questions)

(items.jsonl does not store `trace`; detail counted from live `query.answer` traces.)

- live relation counts (30): `{'cannot_determine': 30}`
- detail counts (30): `{'not_found': 29, 'simultaneous': 1}`
- grounding Acc@1 (cited∩gold ≥50%): **1/30 = 0.033**

| id | stratum | gold | pred | detail | hit | trace (abbrev) |
|----|---------|------|------|--------|-----|----------------|
| sq0001 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Brann Rook shared the bread': not found · grounding 'Nessa Pelt crossed the bridge at Ashmoor': not found ·  |
| sq0002 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Perrin Bramble spoke the vow': not found · grounding 'Thane Harth crossed the bridge at Ashmoor': not found  |
| sq0003 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Perrin Bramble mended the roof at Fairhollow': not found · grounding 'Selka Marrow drew the map in another p |
| sq0004 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Kale Wold drew the map in another part of Saltmarsh': not found · grounding 'Brann Rook opened the gate at M |
| sq0005 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Brann Cinder lit the beacon at Glassport': not found · grounded 'Wynn Tarn mended the roof in another part o |
| sq0006 | unordered | cannot_determine | cannot_determine | simultaneous | True | parsed qtype=order · grounded 'Perrin Bramble drew the map in another part of Fairhollow' -> ev_9 (The description matches exactly with cand |
| sq0007 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Perrin Bramble planted the orchard': not found · grounding 'Vesper Marrow called the council at Ironvale': n |
| sq0008 | unordered | cannot_determine | cannot_determine | not_found | False | parsed qtype=order · grounding 'Vesper Marrow sealed the letter at Fairhollow': not found · grounding 'Kale Pelt sounded the horn in another |
| sq0009 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Birth of Lira Bramble': not found · grounding 'Birth of Selka Marrow': not found · detail=not_found |
| sq0010 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Birth of Selka Marrow': not found · grounding 'Birth of Brann Rook': not found · detail=not_found |
| sq0011 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Birth of Brann Rook': not found · grounding 'Birth of Vesper Rook': not found · detail=not_found |
| sq0012 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Birth of Vesper Rook': not found · grounding 'Birth of Kale Tarn': not found · detail=not_found |
| sq0013 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Birth of Kale Tarn': not found · grounding 'Birth of Nessa Vellum': not found · detail=not_found |
| sq0014 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Birth of Nessa Vellum': not found · grounding 'Vesper Marrow shared the bread': not found · detail=not_found |
| sq0015 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Vesper Marrow shared the bread': not found · grounding 'Kale Wold planted the orchard': not found · detail=n |
| sq0016 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Kale Wold planted the orchard': not found · grounding 'Mira Kestrel broke the silence': not found · detail=n |
| sq0017 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Mira Kestrel broke the silence': not found · grounding 'Selka Marrow left the harbor': not found · detail=no |
| sq0018 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Selka Marrow left the harbor': not found · grounding 'Nessa Vellum broke the silence': not found · detail=no |
| sq0019 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Nessa Vellum broke the silence': not found · grounding 'Joric Harth planted the orchard': not found · detail |
| sq0020 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Joric Harth planted the orchard': not found · grounding 'Brann Cinder sealed the letter': not found · detail |
| sq0021 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Brann Cinder sealed the letter': not found · grounding 'Thane Harth left the harbor': not found · detail=not |
| sq0022 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Thane Harth left the harbor': not found · grounding 'Brann Cinder spoke the vow': not found · detail=not_fou |
| sq0023 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Brann Cinder spoke the vow': not found · grounding 'Nessa Pelt broke the silence': not found · detail=not_fo |
| sq0024 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Nessa Pelt broke the silence': not found · grounding 'Kale Wold raised the banner': not found · detail=not_f |
| sq0025 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Kale Wold raised the banner': not found · grounding 'Mira Harth sounded the horn': not found · detail=not_fo |
| sq0026 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Mira Harth sounded the horn': not found · grounding 'Perrin Bramble raised the banner': not found · detail=n |
| sq0027 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Perrin Bramble raised the banner': not found · grounding 'Kale Wold sounded the horn': not found · detail=no |
| sq0028 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Kale Wold sounded the horn': not found · grounding 'Lira Bramble found the token': not found · detail=not_fo |
| sq0029 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Lira Bramble found the token': not found · grounding 'Mira Wold left the harbor': not found · detail=not_fou |
| sq0030 | aligned | before | cannot_determine | not_found | False | parsed qtype=order · grounding 'Mira Wold left the harbor': not found · grounding 'Nessa Vellum sealed the letter': not found · detail=not_f |

## 6. v1 build + sample answers

- postgres `events` rows: **7**
- last v1 job: events=7 major=7 windows=1
- Neo4j `:Event` nodes for doc: **7**

### 5 v1 answers

**sq0001** stratum=unordered gold=`cannot_determine` pred=`cannot_determine` conf=0.0
- Q: Did Meanwhile, in another part of Fairhollow, Brann Rook shared the bread [E0013] happen before or after At the same hour, Nessa Pelt crossed the bridge at Ashm
- A: The provided events do not include any mention of Brann Rook sharing bread or Nessa Pelt crossing a bridge at Ashmoor. Therefore, it is not possible to determine the order of these events based on the supplied material.

**sq0002** stratum=unordered gold=`cannot_determine` pred=`cannot_determine` conf=0.0
- Q: Did Meanwhile, in another part of Glassport, Perrin Bramble spoke the vow [E0027] happen before or after At the same hour, Thane Harth crossed the bridge at Ash
- A: The events provided do not include any mention of Perrin Bramble speaking a vow or Thane Harth crossing a bridge at Ashmoor. Therefore, it is not possible to determine the order of these events based on the supplied mate

**sq0003** stratum=unordered gold=`cannot_determine` pred=`cannot_determine` conf=0.0
- Q: Did At the same hour, Perrin Bramble mended the roof at Fairhollow [E0015] happen before or after Meanwhile, in another part of Mistfall, Selka Marrow drew the 
- A: The events provided do not include any mention of Perrin Bramble mending the roof at Fairhollow or Selka Marrow drawing a map in Mistfall. Therefore, I cannot determine the order of these events.

**sq0004** stratum=unordered gold=`cannot_determine` pred=`cannot_determine` conf=0.0
- Q: Did Meanwhile, in another part of Saltmarsh, Kale Wold drew the map [E0018] happen before or after At the same hour, Brann Rook opened the gate at Mistfall [E01
- A: The provided events do not include any mention of Kale Wold drawing a map in Saltmarsh or Brann Rook opening a gate at Mistfall. Therefore, it is not possible to determine the order of these events based on the supplied 

**sq0005** stratum=unordered gold=`cannot_determine` pred=`cannot_determine` conf=0.0
- Q: Did At the same hour, Brann Cinder lit the beacon at Glassport [E0011] happen before or after Meanwhile, in another part of Fairhollow, Wynn Tarn mended the roo
- A: The provided events do not include any mention of Brann Cinder lighting a beacon at Glassport or Wynn Tarn mending a roof in Fairhollow. Therefore, it is not possible to determine the order of these events based on the g

## Root cause (diagnosis)

1. **Not exceptions**: `summary.errors=0` for both runs; pipelines return successful `cannot_determine` (coverage 0).
2. **v1**: only a handful of events extracted for the whole synth book; answers say the provided events do not mention the asked characters → universal abstention.
3. **v2**: offsets healthy (quote match 1.000), graph has 15 edges, but **build-time gold span recall = 0.039** and occurs non-telling = **6**. Raw LLM extract cache emitted only **13** mentions for ~150 story events. Grounding is mostly `not_found` → `cannot_determine`.
4. **Shared failure mode**: extract under-recall on dense one-event-per-paragraph text (LLM returns a handful of mentions per window), not offset corruption and not metric bugs.

