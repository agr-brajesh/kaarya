"""FastAPI surface. Thin on purpose -- all logic lives in api_core / engine /
agents / audit. If FastAPI will not install, `python serve.py` gives you the
same routes on the stdlib.

Run: uvicorn app:api --reload   ->  http://127.0.0.1:8000

Note: no auth. Bind to localhost only (uvicorn's default). Do not expose this
to a network -- the endpoints accept and score arbitrary candidate text.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import api_core

api = FastAPI(title="Kaarya")


class Profile(BaseModel):
    text: str = ""
    role_code: str = "13-1111.00"
    limitations: list[str] = []
    # pedigree: stored for the audit trail, never scored
    name: str = ""
    institution: str = ""
    gap_months: int = 0
    city: str = ""
    city_tier: int = 1


@api.get("/api/health")
def health():
    return api_core.health()


@api.post("/api/match")
def api_match(p: Profile):
    return api_core.do_match(p.model_dump())


@api.post("/api/skills/discover")
def api_discover(body: dict):
    """Run Skills Discovery Agent on multi-source profile data."""
    return api_core.do_discover_skills(body)


@api.post("/api/audit")
def api_audit(p: Profile):
    """The twin test. This is the endpoint the demo clicks."""
    return api_core.do_audit(p.model_dump())


@api.post("/api/promote")
def api_promote(p: Profile, substring: str, source: str = "micro-task"):
    """Raise evidence to `demonstrated` after a micro-task passes."""
    return api_core.do_promote(p.model_dump(), substring, source)


@api.get("/api/jd/{code}")
def api_jd(code: str):
    return api_core.do_jd(code)


@api.get("/", response_class=HTMLResponse)
def index():
    return api_core.index_html()
