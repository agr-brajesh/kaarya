# Kaarya — implementation guide + 3-day build plan

Task-level hiring engine that proves its own fairness. Match a person to the
**tasks** of a job rather than the title, name the accommodation where a task
is blocked, carve the requisition where no accommodation exists, and prove no
pedigree attribute moved the ranking.

**Day 1 of your 3 days is already done.** Everything below runs now, offline,
with no API key and no paid service. Read "What is already built", verify it on
your machine in 5 minutes, then start at Day 1 Hour 4 of the plan.

---

## Run it in 60 seconds

```bash
cd kaarya
pip install numpy                # the only hard dependency
python test_kaarya.py            # 27 checks -- run this before every demo
python demo.py                   # the whole narrative, printed
python serve.py                  # UI at http://127.0.0.1:8000  (stdlib only)
```

FastAPI path, once you have network: `pip install -r requirements.txt` then
`uvicorn app:api --reload`. Both servers expose identical routes because both
are thin wrappers over `api_core.py`. `serve.py` exists because pip is the
thing most likely to fail you on demo day.

Binds to `127.0.0.1` with no authentication — it is a local demo server. Do not
put it on a network or a shared machine.

---

## What is already built

| File | What it does | Verified |
|---|---|---|
| `engine.py` | Embeddings (MiniLM + offline fallback), accessibility rules, accommodation KB, `match()`, naive-ATS baseline, SQLite | yes |
| `audit.py` | Twin test, adverse-impact (4/5ths) with a reliability flag, shortlist concentration | yes |
| `agents.py` | Evidence extraction + tier promotion, JD scan/rewrite, phantom requirements, requisition carving | yes |
| `seed.py` | 2 roles with 27 task statements, 2 personas, 200-candidate pool | yes |
| `api_core.py` | All request handling, shared by both servers | yes |
| `serve.py` / `app.py` | Zero-dependency server / FastAPI server | `serve.py` verified end to end; `app.py` needs FastAPI installed |
| `static/index.html` | Single-page UI: task table, carving, twin test | renders, 200 OK |
| `onet_import.py` | Loads the real O*NET text release into the same schema | yes, against a fixture |
| `test_kaarya.py` | 27 checks, each protecting a claim you will make on stage | 27/27 pass |
| `demo.py` | The narrated end-to-end run | yes |

Numbers it currently produces (offline fallback embedder, 200-candidate pool):

- Twin test — Kaarya flip rate **0%**, naive **100%**. All attributes flipped:
  Kaarya **#56 → #56**, naive **#38 → #198**.
- Adverse impact, top-20 — Kaarya **0.94** (passes), naive **0.15** (fails),
  both flagged reliable.
- Deaf candidate → support role — score **51.4**, role **85.4% accessible**,
  captioning named, 2 phone-queue tasks honestly unresolvable, carving
  recommends reassigning **14.6%** of role weight for **100%** coverage.
- JD rewrite — 4 exclusions found, rewrite reads as clean prose.

---

## The one thing to fix first

The pipeline is running the **offline `hash-fallback` embedder**, because the
machine this was built on had no PyPI access. It matches on stemmed words, not
meaning, so lexically distant pairs — "Reconcile transaction records between
source systems and the ledger" vs "reconciled vendor invoices" — fall to `gap`
and absolute scores read low (Meena: 24.4). The *mechanism* is correct; the
similarity is weak.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers==3.3.1
python -c "from engine import embedder; print(embedder().name)"   # all-MiniLM-L6-v2
python calibrate.py      # you write this on Day 1, Hour 5
```

Then re-check `THRESHOLDS["all-MiniLM-L6-v2"] = (0.50, 0.34)` in `engine.py`.
Those two numbers are an **estimate, not measured** — they decide `covered` vs
`coachable` vs `gap`, so every score on your slides depends on them. Calibrate
before you screenshot anything.

---

## 3-day plan, hour by hour

Assumes 4 people and ~12 working hours a day. Solo fallback after each day.
**A = matching/engine, B = agents/LLM, C = frontend, D = data/demo.**

### Day 1 — make it real, not a mock

| Hr | A | B | C | D |
|---|---|---|---|---|
| 0–1 | Everyone: run `test_kaarya.py` + `demo.py` on your own machine. Read `engine.match()` together. **Freeze the JSON contract** — C and D must never wait on A again. |
| 1–3 | Install torch CPU + sentence-transformers; confirm `embedder().name` flips | Get a Groq key, set `GROQ_API_KEY`, confirm `extract_claims` takes the LLM path | Fork `static/index.html` into your React/Vite app, or keep the single file — it already works | Download the O*NET text release, run `onet_import.py <dir> all` |
| 3–5 | Write `calibrate.py`: 30 hand-labelled (claim, task, should-be-covered) pairs, sweep both cutoffs, print the pair that maximises agreement | Write the JD-scan LLM path (regex list stays as the fallback) | Task table with the five verdict colours + keyboard focus states | Pick 6 target occupations; check every one has ≥12 tasks after import |
| 5–7 | Land the calibrated thresholds; re-run `test_kaarya.py` — **it must stay 27/27** | Micro-task flow: 3 short tasks per role that promote a claim to `demonstrated` | Twin-test screen: the two-column naive-vs-Kaarya rank move. **This is the hero shot** | Rebuild the 200-pool over the real O*NET roles; re-check adverse impact |
| 7–9 | Add ability-demand blocking: read the `demands` table so `hearing LV ≥ 3.5` blocks a task even when no keyword fires | Expand `ACCOMMODATIONS` from 11 to ~40 from public Job Accommodation Network guidance | Carving screen: reassign vs drop, coverage-after on both | 3 personas with written scripts: first-gen, deaf, returner |
| 9–12 | Buffer + review each other's diffs | | | |

**End of Day 1 must be true:** real embeddings, calibrated thresholds, real
O*NET tasks, 27/27 checks passing, twin test visible in a browser.

**Solo fallback:** hours 0–1, 1–3 (embeddings only), 3–5 (calibrate), 5–7 (twin
screen). Skip ability-demand blocking and keep the 11 accommodations.

### Day 2 — the two things nobody else will have

| Hr | A | B | C | D |
|---|---|---|---|---|
| 0–2 | Quarantine gate: when `twin_test` returns `quarantined`, the API must refuse to serve a ranking and return the review payload instead | Human-in-the-loop: every carve/rewrite is a proposal with an approve/reject/edit record | HR diff view: original JD on the left, rewrite on the right, approve per line | Mock SuccessFactors OData v2 payloads (`JobRequisition`, `Candidate`, `PerPerson`) from the free Business Accelerator Hub shapes |
| 2–4 | Audit log: append-only JSONL of every decision — input hash, verdict, thresholds, embedder name, timestamp | Learning pathway: derive it from `coachable` + `gap` tasks only, never from a title | Accessibility pass: NVDA or Narrator through the whole flow, then axe DevTools until zero violations | `POST /api/sap/requisition` reading the mock payload end to end |
| 4–6 | Break-glass: a human can override any verdict, and the override is logged with the reason | Phantom-requirement screen: which stated requirements gate zero tasks | Empty/error/loading states — a judge will type nonsense into your textarea | Third role from a different family (warehouse or field) so accessibility rules visibly differ |
| 6–8 | Everyone: **first full rehearsal**, timed, no fixing. Write down every stumble. |
| 8–10 | Fix only the stumbles from the rehearsal. Nothing else. |
| 10–12 | Slides: 6 max — problem, mechanism, twin test, carving, architecture, what's next |

**End of Day 2 must be true:** quarantine works, HR approves a diff, screen
reader gets through it, one SAP-shaped call succeeds, one timed rehearsal done.

**Solo fallback:** quarantine gate, HR diff view, accessibility pass,
rehearsal. Skip the audit log, the third role and the SAP mock — say "next
step" for those and mean it.

### Day 3 — make it survive a hostile room

| Hr | Everyone |
|---|---|
| 0–2 | Judge-mode toggle: a form where a judge edits `NAIVE_RULES` and re-runs the twin test live. This is the single highest-leverage thing left — it converts "you tuned the baseline" from an accusation into a demo beat |
| 2–4 | Failure drills: no network, no API key, empty input, 5000-word input, unknown role code, two browsers at once. Fix what breaks |
| 4–5 | Rehearsal 2, timed. Then freeze the code. Tag it. Zip it |
| 5–6 | Record a 3-minute screen capture of the working demo. If the laptop dies on stage you still have a demo |
| 6–8 | Q&A drill: each person answers the six questions below cold, out loud |
| 7–8 | `README` + repo hygiene: one command to run, comments explaining *why*, no dead files |
| 8–10 | Rehearsal 3 with the slides. Trim to the time limit minus 60 seconds |
| 10–12 | Sleep. Do not add a feature after Hour 5 |

**Freeze rule:** after Day 3 Hour 5, the only permitted commits are text and
bug fixes. Every hackathon loss you will hear about came from a feature added
in the last four hours.

---

## Cut list — drop in this order when you fall behind

1. Third occupation family
2. Audit log JSONL (keep the in-memory decision record)
3. Micro-task flow (hard-code one `demonstrated` claim and say so)
4. SAP OData mock (show the payload shape on a slide instead)
5. Ability-demand blocking (keyword rules already work)
6. React (`static/index.html` is a working UI — shipping it is not a downgrade)

**Never cut:** the twin test, the carving screen, `test_kaarya.py`, the
accessibility pass. Those four *are* the submission.

---

## Demo-day checklist

```
[ ] python test_kaarya.py            -> 27/27
[ ] embedder().name                  -> all-MiniLM-L6-v2, not hash-fallback
[ ] laptop on charge, wifi off once  -> demo still runs
[ ] backup video on the desktop
[ ] serve.py works with pip broken
[ ] browser zoom 100%, font size up for the back of the room
[ ] one tab, one terminal, nothing else open
```

## Six questions you will be asked

**"Your baseline is a straw man."** Hand them the keyboard — `NAIVE_RULES` is
editable and the twin test re-runs live. The claim is narrow: any scorer whose
features include pedigree can be moved by pedigree, and ours cannot, because
those fields never enter `match()`. `test_kaarya.py` asserts that by inspecting
the source.

**"Where is your training data?"** There isn't any, deliberately. No public
dataset pairs protected attributes with hiring outcomes, so we test
counterfactually — copy a candidate, flip one attribute, re-rank. The test
manufactures its own evidence, which is also why it runs on a laptop.

**"Isn't carving just lowering the bar?"** The opposite: we report task coverage
*after* the carve. Reassigning 14.6% of role weight to a pool that already
covers it gives 100% coverage. Dropping the tasks gives 85.4% — we show that
number and refuse to recommend it.

**"Does 0% flip rate mean you're fair?"** No. It means pedigree cannot move our
ranking. It says nothing about whether our task statements or accommodation
rules are right, which is why every rule is a readable list and every carve
needs human approval.

**"Why only 4 agents when the brief lists 7?"** Seven shallow agents is a demo
of a diagram. We built the two that nobody else will have working and made them
survive their own audit.

**"Regulation?"** Hiring is an EU AI Act Annex III high-risk use case. The
Digital Omnibus deferred the standalone high-risk obligations to **2 Dec 2027**;
Article 50 transparency obligations applied from 2 Aug 2026. *Re-verify before
you say this out loud — it was last checked 2026-09-03 and this timeline has
already moved once.*

---

## Honest limitations

- Thresholds `(0.50, 0.34)` for MiniLM are unmeasured. Calibrate them.
- The accommodation KB is 11 entries, curated by hand. It is a demonstration of
  a structure, not clinical guidance.
- `TASK_DOMAIN_RULES` is keyword matching. It is deliberately crude so an
  auditor can read every rule, and it will miss cases.
- `NAIVE_RULES` weights are a stated model of credential filtering, not a
  measurement of any real vendor. Say this before a judge says it for you.
- All personas are synthetic. No real candidate data, no real disability
  disclosures — do not swap in real people for the demo.
- `app.py` (FastAPI) is unverified here because pip had no network; `serve.py`
  is verified end to end over HTTP. Test the FastAPI path on your own machine.



