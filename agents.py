"""Agents. Every LLM call has an offline fallback, so the whole pipeline runs
with zero API keys. Set GROQ_API_KEY to switch the smart path on.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from engine import (ACCOMMODATIONS, Role, SkillClaim, cos, domains_for_task,
                    embedder)

# --------------------------------------------------------------- LLM plumbing

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("KAARYA_MODEL", "llama-3.3-70b-versatile")


def llm(prompt: str, system: str = "Reply with JSON only.") -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    body = json.dumps({
        "model": MODEL, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        GROQ_URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:                      # noqa: BLE001
        print(f"[llm] falling back to offline path: {e}")
        return None


# ------------------------------------------------------- 1. Evidence Agent

ACTION_VERBS = ("handled", "built", "reconcil", "managed", "resolved", "wrote",
                "prepared", "tracked", "trained", "supported", "audited", "fixed",
                "documented", "tested", "designed", "answered", "processed",
                "verified", "escalated", "maintained", "coordinated", "analysed",
                "analyzed", "created", "updated", "reviewed", "operated")

EXTRACT_PROMPT = """Extract concrete work-skill statements from this self-description.
Rules: keep the person's own wording; one action per statement; ignore any mention of
college, employer prestige, gender, city or career gaps.
Return {"claims": [{"statement": str, "tier": "self-declared"}]}.

TEXT:
%s"""


def _segment(text: str) -> list[str]:
    """Cut a free-text blurb into one-action clauses.

    This matters far more than it looks. match() compares every claim to every
    ~12-word task statement, so one 40-word blob has a diluted cosine against
    all of them and lands on `gap` across the board -- we saw a live profile
    score 9.3 instead of 51.4 for exactly this reason. People write these lists
    comma-separated ("handled tickets, updated the CRM, escalated disputes"),
    so cut at hard punctuation first and then wherever a new action verb starts.
    """
    out: list[str] = []
    for chunk in (c for c in re.split(r"[.;\n]", text) if c.strip()):
        words = chunk.split()
        starts = [i for i, w in enumerate(words)
                  if any(w.strip(",").lower().startswith(v) for v in ACTION_VERBS)]
        if len(starts) <= 1:
            out.append(chunk.strip())
            continue
        # keep any lead-in ("for two years I handled...") on the first clause
        bounds = starts if starts[0] == 0 else [0] + starts[1:]
        for a, b in zip(bounds, bounds[1:] + [len(words)]):
            seg = re.sub(r"\s+(and|or|then|plus)$", "",
                         " ".join(words[a:b]).strip(" ,;"))
            if seg:
                out.append(seg)
    return out


def extract_claims(text: str) -> list[SkillClaim]:
    """Skills Discovery. LLM if a key exists, else clause-splitting fallback."""
    out = llm(EXTRACT_PROMPT % text)
    if out:
        try:
            return [SkillClaim(statement=c["statement"],
                               tier=c.get("tier", "self-declared"))
                    for c in json.loads(out)["claims"]][:12]
        except Exception:                        # noqa: BLE001
            pass
    claims = [s for s in _segment(text)
              if len(s.split()) >= 3 and any(v in s.lower() for v in ACTION_VERBS)]
    if not claims:      # no verb we recognise: fall back to punctuation clauses
        claims = [s.strip() for s in re.split(r"[.;,\n]", text)
                  if len(s.split()) >= 4]
    seen: set[str] = set()
    uniq = [s for s in claims if not (s.lower() in seen or seen.add(s.lower()))]
    return ([SkillClaim(statement=s) for s in uniq[:12]]
            or [SkillClaim(statement=text.strip()[:200])])


def promote(claims: list[SkillClaim], statement_substr: str, tier: str,
            source: str) -> list[SkillClaim]:
    """Raise a claim's evidence tier after a micro-task or artifact check.
    Nothing reaches `demonstrated` without passing through here."""
    for c in claims:
        if statement_substr.lower() in c.statement.lower():
            c.tier, c.source = tier, source
    return claims


# --------------------------------------------- 2. Requisition Agent (JD side)

# Each pattern carries the reason and a replacement. Reason strings go straight
# into the HR-facing diff, so write them as you want HR to read them.
EXCLUSIONARY = [
    (r"\b(fresh(er|ers)?|recent graduate|batch of 20\d\d)\b",
     "Age/experience proxy - excludes career returners and displaced workers",
     "open to candidates at this skill level regardless of graduation year"),
    (r"\b(no career gaps?|continuous employment|unbroken)\b",
     "Directly penalises caregiving and medical breaks", ""),
    (r"\b(tier[- ]?1|iit|nit|iim|premier institute|top college)\b",
     "Institution proxy - gates on pedigree, not ability", ""),
    (r"\b(high energy|go[- ]getter|hustle|work hard,? play hard)\b",
     "Culture-coded language shown to deter disabled and older applicants", ""),
    (r"\b(native|mother tongue) (english|speaker)\b",
     "Language nativeness is not a job requirement", "professional working proficiency"),
    (r"\b(must be able to (stand|walk|lift)|physically fit|able[- ]bodied)\b",
     "Blanket physical requirement - state the actual task instead", ""),
    (r"\bexcellent\s+(verbal|telephone|phone|oral)(\s+and\s+\w+)?\s+communication"
     r"|\bhandle\s+high\s+call\s+volume\b",
     "Voice-only framing - state the outcome, not the channel",
     "communicate clearly with customers in the channel they use"),
    (r"\b(bachelor'?s degree required|graduate degree mandatory|b\.?tech only)\b",
     "Degree gate - check whether it gates any actual task", ""),
]


def scan_jd(jd_text: str) -> list[dict]:
    """Employer Readiness: flag exclusionary lines with a suggested rewrite."""
    findings = []
    for pat, reason, repl in EXCLUSIONARY:
        for m in re.finditer(pat, jd_text, flags=re.I):
            findings.append({"phrase": m.group(0), "span": [m.start(), m.end()],
                             "reason": reason, "suggestion": repl,
                             "action": "replace" if repl else "delete"})
    return findings


def rewrite_jd(jd_text: str, findings: list[dict]) -> str:
    """Produce a readable rewritten JD.

    `replace` swaps the phrase in place. `delete` removes the whole clause the
    phrase sits in, because deleting two words out of prose leaves rubble
    ("tier-1 college, high energy team" -> ", , team"). HR reviews this as a
    diff and approves line by line; nothing is applied automatically.
    """
    # Split but remember the separator that followed each clause, so a kept
    # clause keeps its comma instead of being run together with the next one.
    parts = re.split(r"((?<=[.;])\s+|,\s+)", jd_text)
    clauses = [p for p in parts[::2] if p.strip()]
    seps = parts[1::2] + [""]
    kill, subs = set(), []
    for f in findings:
        for i, c in enumerate(clauses):
            if f["phrase"].lower() in c.lower():
                if f["action"] == "delete":
                    kill.add(i)
                else:
                    subs.append((i, f["phrase"], f["suggestion"]))
                break
    out = []
    for i, c in enumerate(clauses):
        if i in kill:
            continue
        for j, phrase, repl in subs:
            if j == i:
                c = re.sub(re.escape(phrase), repl, c, flags=re.I)
                # "Freshers preferred" -> "<replacement> preferred" reads badly:
                # the qualifier belonged to the phrase we just removed.
                c = re.sub(r"\s+(preferred|only|required|mandatory|a must)\b", "",
                           c, flags=re.I)
        out.append(c.strip() + (", " if seps[i].startswith(",") else " "))
    text = " ".join(" ".join(out).split()).rstrip(" ,;")
    text = re.sub(r"\s+([.;,])", r"\1", text)
    text = re.sub(r"(^|[.;] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text if text.endswith(".") else text + "."



PHANTOM_CUTOFF = 0.30


def phantom_requirements(requirements: list[str], role: Role) -> list[dict]:
    """A requirement that matches no task in the role gates nothing. Those are
    the ones you can delete without touching the work."""
    emb = embedder()
    T = emb.encode([t for t, _ in role.tasks])
    R = emb.encode(requirements)
    sim = cos(R, T)
    out = []
    for i, req in enumerate(requirements):
        best = float(sim[i].max())
        j = int(sim[i].argmax())
        out.append({"requirement": req, "best_task_similarity": round(best, 3),
                    "closest_task": role.tasks[j][0],
                    "phantom": best < PHANTOM_CUTOFF,
                    "note": "Gates no task in this role - safe to remove"
                            if best < PHANTOM_CUTOFF else "Backed by real work"})
    return out


# ---------------------------------------------- 3. Carving (fix the job)

def carve(match_result: dict, shared_pool_exists: bool = True) -> dict:
    """Turn unresolvable access-blocks into a requisition restructure proposal.

    Always reports task coverage next to pool impact. A carve that drops
    coverage below 100% is returned as a tradeoff, never a recommendation.
    """
    blocked = [t for t in match_result["tasks"] if t["verdict"] == "access-blocked"]
    weight = sum(t["weight"] for t in blocked)
    proposals = []
    if not blocked:
        return {"needed": False, "proposals": [], "carved_weight": 0.0,
                "coverage_after": 100.0}
    if weight <= 0.15 and shared_pool_exists:
        proposals.append({
            "type": "reassign",
            "detail": f"Move {len(blocked)} task(s) ({weight*100:.0f}% of role weight) "
                      f"to the existing shared/overflow pool that already covers them.",
            "tasks": [t["task"] for t in blocked],
            "coverage_after": 100.0, "recommended": True})
    else:
        proposals.append({
            "type": "split",
            "detail": f"Split into two roles: the {weight*100:.0f}% of work that needs "
                      f"these {len(blocked)} task(s), and the {100-weight*100:.0f}% that does not.",
            "tasks": [t["task"] for t in blocked],
            "coverage_after": 100.0, "recommended": True})
    proposals.append({
        "type": "drop", "detail": "Drop the blocked tasks outright.",
        "tasks": [t["task"] for t in blocked],
        "coverage_after": round((1 - weight) * 100, 1),
        "recommended": False, "tradeoff": "Work stops being done. Shown for honesty."})
    return {"needed": True, "carved_weight": round(weight, 4),
            "accessible_before": match_result["accessible_pct"],
            "accessible_after": 100.0, "proposals": proposals}



