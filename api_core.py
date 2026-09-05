"""Shared API core. Both app.py (FastAPI) and serve.py (stdlib) are thin
wrappers over this module, so the two servers can never drift apart.

Keeping the logic here also means the HTTP surface is testable without
installing anything -- run `python serve.py` and curl it.
"""
from __future__ import annotations

from pathlib import Path

import agents
import audit
import seed
from engine import Candidate, embedder, list_roles, load_role, match

CON = seed.build_db()
POOL = seed.make_pool(200, seed=11)

DEFAULTS = {
    "text": "", "role_code": "13-1111.00", "limitations": [], "name": "",
    "institution": "", "gap_months": 0, "city": "", "city_tier": 1,
}


def clean(body: dict | None) -> dict:
    """Coerce an untrusted JSON body into the fields we accept. Anything the
    client sends that is not in DEFAULTS is dropped -- no surprise kwargs."""
    body = body or {}
    p = dict(DEFAULTS)
    for k, default in DEFAULTS.items():
        v = body.get(k, default)
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = default
        elif isinstance(default, list):
            v = [str(x) for x in v] if isinstance(v, list) else []
        else:
            v = str(v or "")
        p[k] = v
    return p


def to_candidate(p: dict) -> Candidate:
    return Candidate(
        id="live", claims=agents.extract_claims(p["text"]),
        limitations=p["limitations"], name=p["name"],
        institution=p["institution"], gap_months=p["gap_months"],
        city=p["city"], tier_of_city=p["city_tier"])


# ------------------------------------------------------------------ handlers

def health() -> dict:
    return {"embedder": embedder().name, "roles": list_roles(CON),
            "pool_size": len(POOL),
            "warning": None if embedder().name != "hash-fallback" else
            "Running the offline fallback embedder. pip install "
            "sentence-transformers for real semantic matching before you demo."}


def do_match(body: dict) -> dict:
    p = clean(body)
    role, cand = load_role(CON, p["role_code"]), to_candidate(p)
    m = match(cand, role)
    return {"role": role.title, "match": m, "carving": agents.carve(m),
            "claims": [c.__dict__ for c in cand.claims]}


def do_audit(body: dict) -> dict:
    """The twin test. This is the endpoint the demo clicks."""
    p = clean(body)
    return audit.twin_test(to_candidate(p), load_role(CON, p["role_code"]), POOL)


def do_discover_skills(body: dict) -> dict:
    """Run full Skills Discovery Agent across multi-source profile data."""
    profile_data = body.get("profile") or body
    return agents.discover_skills(profile_data, include_adjacent=True)


def do_promote(body: dict, substring: str, source: str = "micro-task") -> dict:
    """Raise evidence to `demonstrated` after a micro-task passes."""
    p = clean(body)
    cand = to_candidate(p)
    agents.promote(cand.claims, substring, "demonstrated", source)
    return {"match": match(cand, load_role(CON, p["role_code"])),
            "claims": [c.__dict__ for c in cand.claims]}


def do_jd(code: str) -> dict:
    role = load_role(CON, code)
    findings = agents.scan_jd(role.jd_text)
    return {"original": role.jd_text, "findings": findings,
            "rewritten": agents.rewrite_jd(role.jd_text, findings),
            "tasks": [{"task": t, "importance": i} for t, i in role.tasks]}


def index_html() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
