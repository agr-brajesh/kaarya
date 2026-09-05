"""Guard rails. Run before every demo: python test_kaarya.py

These are not unit tests for coverage's sake -- each one protects a claim you
are going to make out loud to a judge. If one fails, do not run the demo.
No test framework needed (pip may not work on the day).
"""
from __future__ import annotations

import inspect
import sys

import agents
import audit
import seed
from engine import (Candidate, SkillClaim, load_role, match, naive_score,
                    accommodation_for)

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


CON = seed.build_db()
SUPPORT = load_role(CON, "43-4051.00")
DATAOPS = load_role(CON, "13-1111.00")
POOL = seed.make_pool(200, seed=11)


def test_no_pedigree_in_scorer() -> None:
    """The fairness claim is structural: prove the words are not in the code."""
    src = inspect.getsource(match)
    for field in ("institution", "gap_months", "tier_of_city", "disclosures", "name"):
        check(f"match() never reads .{field}", f".{field}" not in src)


def test_twin_test_direction() -> None:
    """Baseline must not be floor-bound, or the test silently reports no bias."""
    r = audit.twin_test(seed.MEENA, DATAOPS, POOL)
    check("kaarya flip rate is zero", r["flip_rate"]["kaarya"] == 0.0,
          f"kaarya={r['flip_rate']['kaarya']:.0%}")
    check("naive flip rate is non-zero", r["flip_rate"]["naive"] > 0.0,
          f"naive={r['flip_rate']['naive']:.0%}")
    check("privileged baseline has room to fall",
          r["baseline_rank"]["naive"] < len(POOL) * 0.6,
          f"naive baseline #{r['baseline_rank']['naive']} of {len(POOL)}")
    c = r["combined"]
    check("kaarya rank unchanged when all attributes flip",
          c["kaarya"]["rank_privileged"] == c["kaarya"]["rank_as_herself"],
          f"#{c['kaarya']['rank_privileged']} -> #{c['kaarya']['rank_as_herself']}")
    check("naive rank collapses when all attributes flip",
          c["naive"]["rank_as_herself"] > c["naive"]["rank_privileged"],
          f"#{c['naive']['rank_privileged']} -> #{c['naive']['rank_as_herself']}")


def test_evidence_tiers_matter() -> None:
    base = Candidate(id="t", claims=[
        SkillClaim("reconciled vendor invoices and fixed duplicate entries in Excel")])
    low = match(base, DATAOPS)["score"]
    agents.promote(base.claims, "reconciled vendor", "demonstrated", "micro-task")
    high = match(base, DATAOPS)["score"]
    check("demonstrated evidence scores above self-declared", high > low,
          f"{low} -> {high}")


def test_accessibility_routing() -> None:
    a = accommodation_for("hearing", "Join live video calls with the account team")
    check("video call routes to captioning, not the phone queue",
          bool(a) and a["resolves"], a["name"] if a else "none")
    b = accommodation_for("hearing", "Answer the inbound call queue by telephone")
    check("telephone queue is honestly unresolvable",
          bool(b) and not b["resolves"], b["name"] if b else "none")

    m = match(seed.ARJUN, SUPPORT)
    check("deaf candidate is not zeroed out", m["score"] > 40, f"score={m['score']}")
    check("accessible_pct is reported", 0 < m["accessible_pct"] < 100,
          f"{m['accessible_pct']}%")
    check("at least one accommodation is named", bool(m["accommodations_needed"]),
          ", ".join(m["accommodations_needed"]))
    c = agents.carve(m)
    check("carving is triggered and recommends a proposal", c["needed"]
          and any(p["recommended"] for p in c["proposals"]))
    check("drop is offered but never recommended",
          all(not p["recommended"] for p in c["proposals"] if p["type"] == "drop"))


def test_pool_fairness() -> None:
    k = [match(c, DATAOPS)["score"] for c in POOL]
    n = [naive_score(c, DATAOPS) for c in POOL]
    grp = lambda c: "gap" if c.gap_months else "no-gap"    # noqa: E731
    ka = audit.adverse_impact(POOL, k, 20, grp)
    na = audit.adverse_impact(POOL, n, 20, grp)
    check("kaarya passes the 4/5ths rule", ka["passes_four_fifths"],
          f"ratio={ka['adverse_impact_ratio']}")
    check("naive fails the 4/5ths rule", not na["passes_four_fifths"],
          f"ratio={na['adverse_impact_ratio']}")
    check("ratio is flagged reliable at this pool size", ka["reliable"])


def test_jd_rewrite_is_readable() -> None:
    f = agents.scan_jd(SUPPORT.jd_text)
    new = agents.rewrite_jd(SUPPORT.jd_text, f)
    check("JD scan finds the seeded exclusions", len(f) >= 3,
          ", ".join(x["phrase"] for x in f))
    for bad in ("fresher", "tier-1", "high energy"):
        check(f"'{bad}' is gone from the rewrite", bad not in new.lower())
    check("rewrite leaves no dangling punctuation",
          " ," not in new and ", ," not in new and "  " not in new, new)
    check("rewrite keeps the actual work", "customer issues" in new.lower())
    ph = agents.phantom_requirements(
        ["Bachelor's degree in engineering",
         "Experience updating CRM records after customer contact"], SUPPORT)
    check("degree requirement is flagged phantom", ph[0]["phantom"],
          f"sim={ph[0]['best_task_similarity']:.2f}")
    check("real requirement is not flagged phantom", not ph[1]["phantom"],
          f"sim={ph[1]['best_task_similarity']:.2f}")


def test_claim_segmentation() -> None:
    """A judge will type a comma-separated blob. One blob = every task a gap."""
    txt = ("handled customer tickets over email and chat, updated CRM records "
           "after each contact, escalated billing disputes to the finance team")
    claims = agents.extract_claims(txt)
    check("free text splits into one claim per action", len(claims) >= 3,
          f"{len(claims)} claims")
    check("claims stay short enough to match a task statement",
          max(len(c.statement.split()) for c in claims) <= 15)


def test_skills_discovery_agent() -> None:
    """Skills Discovery Agent: multi-source profile, pedigree stripping, and Skill Graph."""
    profile = {
        "text": "IIT Bombay graduate with a 3-year career break. Reconciled vendor invoices and resolved ledger discrepancies in Excel.",
        "micro_credentials": [
            {"name": "Advanced Spreadsheet & Data Cleaning", "issuer": "Coursera",
             "skills": ["data deduplication", "pivot tables"]}
        ],
        "projects": [
            {"title": "Automated Billing Ledger",
             "description": "Built automated spreadsheet models and audited transaction ledgers for small businesses"}
        ],
        "informal_learning": [
            "Self-taught SQL querying through open-source databases"
        ]
    }
    disc = agents.discover_skills(profile, include_adjacent=True)
    check("multi-source profile ingests multiple sources",
          len(disc["sources_analyzed"]) >= 3, f"{len(disc['sources_analyzed'])} sources")
    check("pedigree signals are actively filtered",
          len(disc["pedigree_filtered"]) >= 2, f"{len(disc['pedigree_filtered'])} removed")
    all_stmts = " ".join(c.statement.lower() for c in disc["claims"])
    check("IIT Bombay is stripped from skill claims", "iit" not in all_stmts)
    check("career break length is stripped from skill claims", "3-year" not in all_stmts)
    check("corroborated tier assigned to micro-credentials or projects",
          any(c.tier == "corroborated" for c in disc["claims"]))
    check("knowledge graph infers adjacent capabilities",
          len(disc["adjacent_skills"]) > 0,
          f"inferred: {[a['title'] for a in disc['adjacent_skills']]}")
    adj = disc["adjacent_skills"][0]
    check("adjacent skills carry explainability metadata",
          "explanation" in adj and "inferred_from" in adj and "confidence" in adj)


if __name__ == "__main__":
    from engine import embedder
    print(f"embedder: {embedder().name}\n")
    for fn in (test_no_pedigree_in_scorer, test_twin_test_direction,
               test_evidence_tiers_matter, test_accessibility_routing,
               test_pool_fairness, test_jd_rewrite_is_readable,
               test_claim_segmentation, test_skills_discovery_agent):
        print(fn.__name__)
        fn()
        print()
    print(f"{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
