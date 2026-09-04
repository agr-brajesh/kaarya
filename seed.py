"""Seed data so the pipeline runs before you have O*NET downloaded.

Task statements below are written in O*NET style. Replace with the real thing
via onet_import.py -- the schema is identical, so nothing downstream changes.
"""
from __future__ import annotations

import random

from engine import Candidate, SkillClaim, db

ROLES = {
    "43-4051.00": ("Customer Support Specialist",
                   "Resolve customer issues across channels, maintain account records, "
                   "and escalate unresolved problems. Excellent verbal and telephone "
                   "communication. Freshers preferred, tier-1 college, high energy team.",
                   [("Respond to customer questions received by email and web form.", 5.0),
                    ("Update customer account records in the CRM after each contact.", 4.8),
                    ("Answer inbound telephone calls from the customer support queue.", 4.5),
                    ("Handle high call volume during peak hours on the phone queue.", 4.2),
                    ("Investigate billing discrepancies using internal reports.", 4.4),
                    ("Write summaries of recurring issues for the product team.", 4.0),
                    ("Escalate unresolved complaints to the relevant specialist team.", 4.3),
                    ("Prepare a weekly report of resolution times and backlog.", 3.8),
                    ("Verify customer identity before releasing account information.", 4.1),
                    ("Document new troubleshooting steps in the internal knowledge base.", 3.6),
                    ("Coordinate refunds with the finance team.", 3.5),
                    ("Join live video calls with enterprise customers to review issues.", 3.4),
                    ("Track service level agreement adherence for assigned tickets.", 3.9),
                    ("Test new support tooling and report defects.", 3.0),
                    ("Train new team members on the ticketing workflow.", 3.2)]),
    "13-1111.00": ("Data Operations Analyst",
                   "Own recurring data pipelines and reconciliation. Bachelor's degree "
                   "required. Must be able to stand and walk the operations floor.",
                   [("Reconcile transaction records between source systems and the ledger.", 5.0),
                    ("Identify and fix duplicate records in incoming data files.", 4.9),
                    ("Build spreadsheet models to summarise monthly volumes.", 4.6),
                    ("Query the reporting database to answer stakeholder questions.", 4.7),
                    ("Document data quality rules and exceptions.", 4.0),
                    ("Prepare month-end reconciliation reports for finance.", 4.5),
                    ("Investigate the root cause of failed data loads.", 4.3),
                    ("Maintain the schedule of recurring data refresh jobs.", 3.9),
                    ("Verify currency and unit conversions in vendor files.", 4.2),
                    ("Coordinate with vendors to correct malformed submissions.", 3.7),
                    ("Review dashboards for anomalies before publication.", 3.8),
                    ("Write handover notes for the operations runbook.", 3.4)]),
}

# --- personas -------------------------------------------------------------

MEENA_TEXT = (
    "I reconciled vendor invoices every month and fixed duplicate entries in Excel. "
    "I built spreadsheet summaries of monthly payment volumes. "
    "I coordinated with vendors when their files were wrong. "
    "I prepared month-end payment reports for my manager. "
    "I verified currency amounts on international invoices."
)

MEENA = Candidate(
    id="meena",
    name="Meena Devi", institution="Government Polytechnic, Buxar",
    gap_months=26, city="Buxar", tier_of_city=3,
    claims=[SkillClaim(s) for s in [
        "reconciled vendor invoices every month and fixed duplicate entries in Excel",
        "built spreadsheet summaries of monthly payment volumes",
        "coordinated with vendors when their files were wrong",
        "prepared month-end payment reports for my manager",
        "verified currency amounts on international invoices",
    ]],
)

ARJUN = Candidate(
    id="arjun",
    name="Arjun Sharma", institution="Anna University", city="Coimbatore", tier_of_city=2,
    limitations=["hearing"],
    claims=[SkillClaim(s) for s in [
        "responded to customer questions over email and live chat",
        "updated customer account records in the CRM after every contact",
        "investigated billing discrepancies using internal reports",
        "escalated unresolved complaints to specialist teams",
        "documented troubleshooting steps in the knowledge base",
        "prepared weekly reports of resolution times and backlog",
        "verified customer identity before releasing account details",
    ]],
)

_FIRST = ["Rohit", "Priya", "Sandeep", "Anita", "Vikram", "Neha", "Imran", "Divya"]
_INST = ["IIT Bombay", "NIT Trichy", "VIT Vellore", "Anna University",
         "Government Polytechnic, Buxar", "Kakatiya University"]
_CITY = ["Bengaluru", "Pune", "Hyderabad", "Kochi", "Buxar", "Lucknow"]


def make_pool(n: int = 40, seed: int = 7) -> list[Candidate]:
    """Synthetic pool. Skill text is drawn from a shared bank so that pedigree
    is the only thing that systematically varies -- which is exactly what makes
    the twin test and the adverse-impact numbers interpretable."""
    rnd = random.Random(seed)
    bank = [s.statement for s in MEENA.claims] + [s.statement for s in ARJUN.claims] + [
        "answered inbound phone calls in a support queue",
        "trained new joiners on the ticketing workflow",
        "tested internal tools and reported defects",
        "queried the reporting database for stakeholder questions",
        "maintained the schedule of recurring refresh jobs",
    ]
    pool = []
    for i in range(n):
        k = rnd.randint(2, 6)
        pool.append(Candidate(
            id=f"pool-{i:02d}", name=f"{rnd.choice(_FIRST)} K",
            institution=rnd.choice(_INST), city=rnd.choice(_CITY),
            tier_of_city=rnd.choice([1, 1, 2, 3]),
            gap_months=rnd.choice([0, 0, 0, 6, 18, 26]),
            claims=[SkillClaim(s) for s in rnd.sample(bank, k)]))
    return pool


def build_db():
    con = db()
    con.execute("DELETE FROM occupations"); con.execute("DELETE FROM tasks")
    for code, (title, desc, tasks) in ROLES.items():
        con.execute("INSERT INTO occupations VALUES (?,?,?)", (code, title, desc))
        con.executemany("INSERT INTO tasks VALUES (?,?,?)",
                        [(code, s, imp) for s, imp in tasks])
    con.commit()
    return con


if __name__ == "__main__":
    build_db()
    print(f"seeded {len(ROLES)} roles")

