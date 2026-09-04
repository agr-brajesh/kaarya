"""Kaarya engine: embeddings, task store, task-level matching, accessibility rules.

Design rule that makes the fairness claim true rather than aspirational:
the Kaarya score is computed ONLY from evidenced skill statements matched
against role task statements. College name, career-gap length, candidate
name and city are never features. That is why it survives the twin test in
audit.py, while the naive baseline (which embeds the whole raw profile) does not.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).with_name("kaarya.db")

# ---------------------------------------------------------------- embeddings

DIM = 512


class HashEmbedder:
    """Dependency-free fallback: stemmed word + bigram bag, L2 normalised.

    Good enough to develop and demo the *mechanism*. Install
    sentence-transformers before judging day for real semantic matching.
    """

    name = "hash-fallback"

    # Domain bridges so the lexical fallback can span obvious synonyms. This
    # exists ONLY to make the fallback usable offline; the real MiniLM path
    # does not need it. Keep it small and auditable.
    BRIDGE = {
        "invoic": "transact payment", "transact": "invoic payment",
        "payment": "invoic transact", "ledger": "transact record",
        "duplicat": "dedup", "excel": "spreadsheet", "spreadsheet": "excel",
        "reconcil": "match verify", "quer": "sql databas", "databas": "quer sql",
        "vendor": "supplier", "summar": "report", "report": "summar",
        "ticket": "case issue", "crm": "account record", "escalat": "refer",
    }

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ations", "ation", "ing", "ies", "ied", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        stops = {"the", "and", "for", "with", "from", "that", "this", "each",
                 "every", "into", "when", "their", "they", "was", "were", "are"}
        words = [self._stem(w) for w in text.split() if len(w) > 2 and w not in stops]
        toks = [(w, 1.0) for w in words]
        toks += [(f"{a}_{b}", 0.6) for a, b in zip(words, words[1:])]
        for w in words:                      # synonym bridges, down-weighted
            for key, expansion in self.BRIDGE.items():
                if w.startswith(key):
                    toks += [(e, 0.7) for e in expansion.split()]
        for token, weight in toks:
            h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            v[h % DIM] += weight
        n = np.linalg.norm(v)
        return v / n if n else v


    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])



class SBertEmbedder:
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.m.encode(texts, normalize_embeddings=True)


_EMB = None


def embedder():
    """Real model if installed, hash fallback otherwise. Never crashes."""
    global _EMB
    if _EMB is None:
        try:
            _EMB = SBertEmbedder()
        except Exception:
            _EMB = HashEmbedder()
    return _EMB


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


# ------------------------------------------------------- accessibility model

# Functional domains a task can demand. Keep this list short and defensible.
DOMAINS = ["hearing", "vision", "speech", "dexterity", "mobility", "schedule"]

# Keyword rules: which domain a task statement engages. Crude on purpose --
# every rule is inspectable, which is what an auditor needs.
TASK_DOMAIN_RULES: dict[str, list[str]] = {
    "hearing": ["phone", "call", "verbal", "listen", "hearing", "telephone", "dictat"],
    "speech": ["phone", "call", "present", "speak", "verbal", "negotiat", "interview"],
    "vision": ["inspect", "read", "review document", "proofread", "visual", "monitor screen"],
    "dexterity": ["assemble", "type", "handle", "operate machine", "repair", "install"],
    "mobility": ["travel", "site visit", "walk", "lift", "warehouse", "field"],
    "schedule": ["shift", "on-call", "overtime", "night", "weekend", "rotational"],
}

# Accommodation KB. Curate ~40 of these from public Job Accommodation Network
# guidance. `resolves=False` is the important case: it is what triggers carving.
ACCOMMODATIONS = [
    # --- hearing: order is deliberate, most specific first ---------------
    {"id": "phone-queue", "domain": "hearing", "name": "Inbound voice phone queue",
     "cost": "n/a", "resolves": False, "triggers": r"telephone|phone queue|inbound call",
     "note": "No accommodation makes a synchronous voice queue accessible. Carve it."},
    {"id": "cap-live", "domain": "hearing", "name": "Live captioning on video calls",
     "cost": "low", "resolves": True, "triggers": r"video call|live video|meeting"},
    {"id": "chat-first", "domain": "hearing", "name": "Route customers to chat/email channel",
     "cost": "none", "resolves": True, "default": True},
    # --- vision ----------------------------------------------------------
    {"id": "screen-reader", "domain": "vision",
     "name": "Screen reader + accessible document pipeline",
     "cost": "low", "resolves": True, "default": True},
    {"id": "magnify", "domain": "vision", "name": "Screen magnification / high-contrast theme",
     "cost": "none", "resolves": True, "triggers": r"dashboard|proofread"},
    # --- speech ----------------------------------------------------------
    {"id": "async-present", "domain": "speech",
     "name": "Allow recorded or written updates instead of live presenting",
     "cost": "none", "resolves": True, "triggers": r"present|report"},
    {"id": "stt", "domain": "speech", "name": "Speech-to-text relay",
     "cost": "low", "resolves": True, "default": True},
    # --- dexterity / mobility / schedule ---------------------------------
    {"id": "alt-input", "domain": "dexterity",
     "name": "Alternative input device / voice control",
     "cost": "medium", "resolves": True, "default": True},
    {"id": "remote", "domain": "mobility", "name": "Remote or hybrid work arrangement",
     "cost": "none", "resolves": True, "default": True},
    {"id": "night-shift", "domain": "schedule", "name": "Mandatory rotational night shift",
     "cost": "n/a", "resolves": False, "triggers": r"night|rotational",
     "note": "Cannot be accommodated for a fixed-hours caregiver. Carve or re-staff."},
    {"id": "flex-hours", "domain": "schedule", "name": "Flexible / fixed-daytime hours",
     "cost": "none", "resolves": True, "default": True},
]



def domains_for_task(statement: str) -> list[str]:
    s = statement.lower()
    return [d for d, kws in TASK_DOMAIN_RULES.items() if any(k in s for k in kws)]


def accommodation_for(domain: str, statement: str) -> dict | None:
    """Pick the most specific accommodation for this task.

    Order matters: specific `triggers` win over the domain default, and an
    unresolvable match is what escalates the task to the carving engine
    instead of quietly claiming it was fixed.
    """
    s = statement.lower()
    pool = [a for a in ACCOMMODATIONS if a["domain"] == domain]
    for a in pool:                                   # specific rules first
        trig = a.get("triggers")
        if trig and re.search(trig, s):
            return a
    for a in pool:                                   # then domain default
        if a.get("default"):
            return a
    return pool[0] if pool else None



# ------------------------------------------------------------- data classes

TIER_WEIGHT = {"self-declared": 0.55, "corroborated": 0.8, "demonstrated": 1.0}


@dataclass
class SkillClaim:
    statement: str          # "reconciled vendor invoices and fixed duplicates in Excel"
    tier: str = "self-declared"
    source: str = "self"


@dataclass
class Candidate:
    id: str
    # --- features the Kaarya score is allowed to see -----------------
    claims: list[SkillClaim] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)   # functional domains
    constraints: dict = field(default_factory=dict)        # {"remote_only": True, ...}
    # --- pedigree fields: recorded for the audit, NEVER scored --------
    name: str = ""
    institution: str = ""
    gap_months: int = 0
    city: str = ""
    tier_of_city: int = 1
    disclosures: list[str] = field(default_factory=list)

    def raw_text(self) -> str:
        """What a naive ATS would embed: everything, pedigree included."""
        bits = [self.name, f"{self.institution} graduate", f"based in {self.city}"]
        if self.gap_months:
            bits.append(f"career break of {self.gap_months} months")
        bits += self.disclosures
        bits += [c.statement for c in self.claims]
        return ". ".join(b for b in bits if b)



@dataclass
class Role:
    code: str
    title: str
    tasks: list[tuple[str, float]]          # (statement, importance 1-5)
    jd_text: str = ""

    def task_weights(self) -> np.ndarray:
        w = np.array([imp for _, imp in self.tasks], dtype=np.float32)
        return w / w.sum()


# ---------------------------------------------------------------- matching

# Cutoffs are embedder-specific. Cosine scales differ between a lexical bag and
# a sentence transformer, so tune per model instead of hard-coding one number.
# Re-run calibrate.py after you install sentence-transformers.
THRESHOLDS = {
    "hash-fallback": (0.30, 0.16),
    "all-MiniLM-L6-v2": (0.50, 0.34),
}


def cutoffs() -> tuple[float, float]:
    return THRESHOLDS.get(embedder().name, (0.50, 0.34))


def match(cand: Candidate, role: Role) -> dict:
    """Task-level match. Returns per-task verdicts + a score in [0,100].

    Only cand.claims / cand.limitations / cand.constraints are read here.
    Grep this function for `institution` or `gap_months` -- you will not find
    them. That absence IS the fairness guarantee.
    """
    covered_at, coachable_at = cutoffs()
    emb = embedder()

    task_texts = [t for t, _ in role.tasks]
    T = emb.encode(task_texts)
    if cand.claims:
        C = emb.encode([c.statement for c in cand.claims])
        tiers = np.array([TIER_WEIGHT[c.tier] for c in cand.claims], dtype=np.float32)
        sim = cos(T, C) * tiers            # (tasks, claims), tier-discounted
        best = sim.max(axis=1)
        who = sim.argmax(axis=1)
    else:
        best = np.zeros(len(role.tasks), dtype=np.float32)
        who = np.zeros(len(role.tasks), dtype=int)

    weights, rows = role.task_weights(), []
    for i, (stmt, imp) in enumerate(role.tasks):
        blocking = [d for d in domains_for_task(stmt) if d in cand.limitations]
        acc = accommodation_for(blocking[0], stmt) if blocking else None
        if blocking:
            verdict = "accommodatable" if acc and acc["resolves"] else "access-blocked"
        elif best[i] >= covered_at:
            verdict = "covered"
        elif best[i] >= coachable_at:
            verdict = "coachable"
        else:
            verdict = "gap"
        rows.append({
            "task": stmt, "importance": imp, "weight": round(float(weights[i]), 4),
            "similarity": round(float(best[i]), 3), "verdict": verdict,
            "evidence": cand.claims[who[i]].statement if cand.claims else None,
            "evidence_tier": cand.claims[who[i]].tier if cand.claims else None,
            "blocking_domain": blocking[0] if blocking else None,
            "accommodation": acc,
        })
    return _summarise(rows, weights)


CREDIT = {"covered": 1.0, "accommodatable": 1.0, "coachable": 0.45,
          "access-blocked": 0.0, "gap": 0.0}


def _summarise(rows: list[dict], weights: np.ndarray) -> dict:
    score = sum(CREDIT[r["verdict"]] * r["weight"] for r in rows) * 100
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in CREDIT}
    unresolved = [r for r in rows if r["verdict"] == "access-blocked"]
    access_weight = sum(r["weight"] for r in unresolved)
    return {
        "score": round(score, 1),
        "counts": counts,
        "n_tasks": len(rows),
        "accessible_pct": round((1 - access_weight) * 100, 1),
        "unresolved_tasks": [r["task"] for r in unresolved],
        "accommodations_needed": sorted({
            r["accommodation"]["name"] for r in rows
            if r["accommodation"] and r["accommodation"]["resolves"]}),
        "tasks": rows,
    }


def naive_score(cand: Candidate, role: Role) -> float:
    """The control group: a conventional resume screener.

    Two parts, both of which real systems demonstrably do and which the track
    brief itself names ("keyword matching, credential filtering, urban-centric
    networks"):

      1. whole-document similarity  - embed the entire profile against the
         entire job description, so pedigree words sit inside the features;
      2. credential heuristics       - the explicit screening rules recruiters
         encode: prestige-institution boost, career-gap penalty, metro boost.

    BE HONEST ABOUT THIS ON STAGE. These weights are a stated model of
    credential filtering, not a measurement of any specific vendor, and they
    were not tuned to make Kaarya look good -- they live in NAIVE_RULES so a
    judge can change them and re-run. The claim is narrow and defensible:
    *any* scorer whose features include pedigree can be moved by pedigree,
    and ours cannot, because those fields never enter match().
    """
    emb = embedder()
    v = emb.encode([cand.raw_text(), role.jd_text or " ".join(t for t, _ in role.tasks)])
    s = float(v[0] @ v[1]) * 100
    r = NAIVE_RULES
    if any(k in cand.institution.lower() for k in r["prestige_tokens"]):
        s += r["prestige_bonus"]
    if cand.gap_months:
        s -= r["gap_penalty_per_year"] * (cand.gap_months / 12)
    if cand.tier_of_city == 1:
        s += r["metro_bonus"]
    if cand.disclosures:
        s -= r["disclosure_penalty"]
    return s


NAIVE_RULES = {
    "prestige_tokens": ["iit", "nit", "iim", "bits", "vit", "anna university"],
    "prestige_bonus": 6.0,
    "gap_penalty_per_year": 4.0,
    "metro_bonus": 3.0,
    "disclosure_penalty": 2.0,
}



# ------------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS occupations(code TEXT PRIMARY KEY, title TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS tasks(code TEXT, statement TEXT, importance REAL);
CREATE TABLE IF NOT EXISTS demands(code TEXT, element TEXT, scale TEXT, value REAL);
CREATE INDEX IF NOT EXISTS ix_tasks_code ON tasks(code);
"""


def db(path: Path | None = None) -> sqlite3.Connection:
    """SQLite next to the code. Override with KAARYA_DB; falls back to a temp
    dir when the working directory is on a filesystem SQLite cannot lock
    (network shares, some container mounts).

    check_same_thread=False is required, not optional: the connection is opened
    once at import and then read from worker threads -- uvicorn runs `def`
    endpoints in a threadpool and ThreadingHTTPServer spawns a thread per
    request. We only ever SELECT after build_db(), so shared reads are safe.
    """
    import os
    import tempfile
    target = Path(path or os.environ.get("KAARYA_DB") or DB_PATH)
    for candidate in (target, Path(tempfile.gettempdir()) / "kaarya.db"):
        try:
            con = sqlite3.connect(candidate, check_same_thread=False)
            con.executescript(SCHEMA)
            return con
        except sqlite3.OperationalError:
            continue
    raise RuntimeError("could not open a SQLite database anywhere")



def load_role(con: sqlite3.Connection, code: str) -> Role:
    title, desc = con.execute(
        "SELECT title, description FROM occupations WHERE code=?", (code,)).fetchone()
    tasks = con.execute(
        "SELECT statement, importance FROM tasks WHERE code=? ORDER BY importance DESC",
        (code,)).fetchall()
    return Role(code=code, title=title, tasks=[(s, i) for s, i in tasks], jd_text=desc)


def list_roles(con: sqlite3.Connection) -> list[dict]:
    return [{"code": c, "title": t} for c, t in
            con.execute("SELECT code, title FROM occupations ORDER BY title")]





