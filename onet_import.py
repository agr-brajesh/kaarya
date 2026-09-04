"""Load the real O*NET bulk download into the same schema seed.py writes.

Why this file exists: the demo runs on seed.py so it never depends on a
download, but a judge will ask "is this real occupational data?". Point at the
free O*NET text release and the answer becomes yes without changing a single
line downstream -- match(), audit.py and the UI only ever see load_role().

    1. https://www.onetcenter.org/database.html -> "Text/tab-delimited"
    2. unzip anywhere, e.g. ./db_29_0_text/
    3. python onet_import.py ./db_29_0_text

Files used (tab-delimited, UTF-8, header row):
    Occupation Data.txt   O*NET-SOC Code | Title | Description
    Task Statements.txt   O*NET-SOC Code | Task ID | Task | Task Type | ...
    Task Ratings.txt      O*NET-SOC Code | Task ID | Scale ID | Data Value | ...
    Abilities.txt         O*NET-SOC Code | Element ID | Element Name | Scale ID | Data Value

Task importance is Task Ratings scale "IM" (1-5). If Task Ratings.txt is
missing we fall back to importance 3.0 for every task, which still works --
task_weights() just becomes uniform.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from engine import db

# Abilities that map onto our functional-limitation domains. Element IDs are
# stable across O*NET releases, which is why we key on them and not on names.
ABILITY_DOMAIN = {
    "1.A.4.b.1": "hearing",     # Hearing Sensitivity
    "1.A.4.b.2": "hearing",     # Auditory Attention
    "1.A.4.b.4": "speech",      # Speech Recognition
    "1.A.4.b.5": "speech",      # Speech Clarity
    "1.A.4.a.1": "vision",      # Near Vision
    "1.A.4.a.2": "vision",      # Far Vision
    "1.A.2.a.2": "dexterity",   # Manual Dexterity
    "1.A.2.a.3": "dexterity",   # Finger Dexterity
    "1.A.3.a.1": "mobility",    # Static Strength
    "1.A.3.c.3": "mobility",    # Gross Body Coordination
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def col(row: dict, *names: str) -> str:
    """O*NET renames columns between releases; accept any of the aliases."""
    for n in names:
        if n in row:
            return row[n].strip()
    return ""


def load(src: Path, codes: list[str] | None = None, max_tasks: int = 25) -> dict:
    """Import occupations, tasks and ability demands. `codes=None` imports all.

    Returns a small report so you can see what landed instead of trusting it.
    """
    con = db()
    keep = set(codes) if codes else None

    def wanted(code: str) -> bool:
        return keep is None or code in keep

    # --- importance ratings, keyed (code, task_id) ------------------------
    imp: dict[tuple[str, str], float] = {}
    for r in rows(src / "Task Ratings.txt"):
        if col(r, "Scale ID") != "IM":
            continue
        code, tid = col(r, "O*NET-SOC Code"), col(r, "Task ID")
        if wanted(code):
            try:
                imp[(code, tid)] = float(col(r, "Data Value"))
            except ValueError:
                pass

    occ = [(col(r, "O*NET-SOC Code"), col(r, "Title"), col(r, "Description"))
           for r in rows(src / "Occupation Data.txt")]
    occ = [o for o in occ if wanted(o[0])]
    if not occ:
        raise SystemExit(f"no occupations matched in {src / 'Occupation Data.txt'}")

    tasks: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows(src / "Task Statements.txt"):
        code = col(r, "O*NET-SOC Code")
        if not wanted(code):
            continue
        stmt = col(r, "Task")
        if not stmt:
            continue
        tasks[code].append((stmt, imp.get((code, col(r, "Task ID")), 3.0)))

    demands: list[tuple[str, str, str, float]] = []
    for r in rows(src / "Abilities.txt"):
        code, eid = col(r, "O*NET-SOC Code"), col(r, "Element ID")
        if not wanted(code) or eid not in ABILITY_DOMAIN:
            continue
        if col(r, "Scale ID") != "LV":          # LV = level required, 0-7
            continue
        try:
            demands.append((code, ABILITY_DOMAIN[eid], "LV", float(col(r, "Data Value"))))
        except ValueError:
            pass

    with con:
        con.executemany("INSERT OR REPLACE INTO occupations VALUES (?,?,?)", occ)
        for code, ts in tasks.items():
            ts.sort(key=lambda t: -t[1])
            con.execute("DELETE FROM tasks WHERE code=?", (code,))
            con.executemany("INSERT INTO tasks VALUES (?,?,?)",
                            [(code, s, i) for s, i in ts[:max_tasks]])
        con.executemany("INSERT OR REPLACE INTO demands VALUES (?,?,?,?)", demands)

    return {"occupations": len(occ),
            "tasks": sum(min(len(v), max_tasks) for v in tasks.values()),
            "demands": len(demands),
            "codes": [c for c, _, _ in occ][:10]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    # Default to the two roles the demo narrates; pass more codes to widen it.
    default = ["43-4051.00", "13-1111.00"]
    picked = sys.argv[2:] or default
    print(load(Path(sys.argv[1]), None if picked == ["all"] else picked))
