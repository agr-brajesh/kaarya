# AGENTS.md — Kaarya

Context for any coding agent working in this repo. Humans: read `README.md`,
which carries the 3-day build plan.

## What this project is

A task-level hiring engine that proves its own fairness. A job is ~15–25 O*NET
task statements with importance weights, not a title. A candidate is a set of
evidenced skill claims. Every task gets a verdict: `covered`, `accommodatable`,
`coachable`, `access-blocked`, or `gap`.

Two mechanics carry the whole thing:

1. **The twin test** (`audit.py`) — copy a candidate, flip exactly one pedigree
   attribute (institution, career gap, name signal, city tier, disability
   disclosure), re-rank against a pool, measure the move. Needs no labelled
   hiring data because it manufactures its own evidence.
2. **Requisition carving** (`agents.carve`) — when a task is blocked and no
   accommodation resolves it, propose reassigning or splitting it and report
   task coverage after the change.

## Commands

```bash
pip install numpy                    # the only hard dependency
python test_kaarya.py                # 27 checks -- run before every commit
python demo.py                       # full narrated end-to-end run
python serve.py                      # stdlib server, http://127.0.0.1:8000
uvicorn app:api --reload             # FastAPI server, same routes
python onet_import.py ./db_29_0_text # load the real O*NET text release
```

Env vars: `KAARYA_DB` (sqlite path), `KAARYA_PORT`, `KAARYA_HOST`,
`GROQ_API_KEY` (optional — every LLM call has an offline fallback),
`KAARYA_MODEL`.

There is no build step and no test framework. `test_kaarya.py` is plain Python
and exits non-zero on failure; wire it into CI as-is.

## Architecture

| File | Role |
|---|---|
| `engine.py` | Embeddings, accessibility rules, accommodation KB, `match()`, `naive_score()`, SQLite schema |
| `audit.py` | Twin test, adverse impact (4/5ths), shortlist concentration |
| `agents.py` | Claim extraction + tier promotion, JD scan/rewrite, phantom requirements, carving |
| `seed.py` | Demo roles, personas, 200-candidate pool, `build_db()` |
| `api_core.py` | All request handling. Both servers are thin wrappers over this |
| `serve.py` / `app.py` | Stdlib server / FastAPI server |
| `onet_import.py` | Real O*NET text release into the same schema |
| `static/index.html` | Single-page UI, no build tooling |
| `test_kaarya.py` | 27 checks, each defending a claim made to a judge |

Dependency direction is one-way: `engine` ← `agents`/`audit`/`seed` ←
`api_core` ← `serve`/`app`. Do not import upward.

## Invariants — breaking these breaks the pitch

- **`match()` must never read `institution`, `gap_months`, `tier_of_city`,
  `name`, or `disclosures`.** That absence *is* the fairness guarantee, and
  `test_no_pedigree_in_scorer` asserts it by inspecting the function's source.
  If you need pedigree for the audit trail, read it outside `match()`.
- **Both servers stay thin.** Logic goes in `api_core.py` or below, so the
  FastAPI and stdlib paths cannot drift.
- **`resolves: False` accommodations are load-bearing.** They are how the
  system says "this cannot be accommodated" instead of quietly claiming a fix.
  Never make an unresolvable entry resolve to keep a score up.
- **`drop` is always offered and never recommended** in carving proposals.
- **`NAIVE_RULES` stays a mutable dict.** A judge must be able to edit the
  baseline's weights and re-run. Do not inline those numbers.
- **`test_kaarya.py` must stay at 27/27.** If a change lowers the count, the
  check it removed was protecting something.

## Bugs already fixed — do not reintroduce

- `sqlite3.connect` needs `check_same_thread=False`. The connection opens once
  at import and is read from worker threads; uvicorn runs `def` endpoints in a
  threadpool and `ThreadingHTTPServer` spawns a thread per request.
- `extract_claims` must split free text into one-action clauses, commas
  included. One 40-word blob has a diluted cosine against every ~12-word task
  statement — a live profile scored 9.3 instead of 22.0 because of this.
- `accommodation_for` checks specific `triggers` regexes before domain
  defaults. Otherwise "join live video calls" matches the unresolvable phone
  queue rule and a deaf candidate is wrongly blocked.
- `rewrite_jd` deletes whole clauses and preserves separators. Span-level
  deletion of two words out of prose leaves rubble.
- `twin_test` starts from `privileged_base()`. A floor-ranked baseline has no
  room to fall, so a biased scorer silently reports no effect.

## Known-unmeasured

`THRESHOLDS["all-MiniLM-L6-v2"] = (0.50, 0.34)` in `engine.py` is an estimate,
never calibrated. It decides `covered` vs `coachable` vs `gap`, so every score
depends on it. If you are asked to improve accuracy, calibrate this first —
before touching any model or prompt.

## Conventions

Python 3.10+, `from __future__ import annotations`, standard library first,
4-space indent, ~88 column soft limit. Comments explain *why*, not *what*;
several existing comments record a specific bug or a defence against a judge's
question — keep them. Type hints on public functions. No new dependencies
without pinning them in `requirements.txt`.

Synthetic personas only. No real candidate data and no real disability
disclosures, ever.
