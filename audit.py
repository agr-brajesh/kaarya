"""Auditor Agent: the twin test, fairness metrics, and the quarantine gate.

The twin test needs no labelled hiring data. It manufactures its own evidence:
copy a candidate, flip exactly ONE pedigree attribute, re-rank, measure the move.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from engine import Candidate, Role, match, naive_score

# Each twin flips one attribute only. Add to this list; it is your test suite.
TWINS: dict[str, dict] = {
    "institution": {"from": "IIT Bombay", "to": "Government Polytechnic, Buxar"},
    "career_gap": {"from": 0, "to": 26},
    "name_signal": {"from": "Arjun Sharma", "to": "Meena Devi"},
    "city_tier": {"from": "Bengaluru", "to": "Buxar"},
    "disability_disclosure": {"from": "", "to": "person with a disability; uses a screen reader"},
}


def make_twin(base: Candidate, attribute: str) -> Candidate:
    t = copy.deepcopy(base)
    t.id = f"{base.id}::twin-{attribute}"
    spec = TWINS[attribute]
    if attribute == "institution":
        t.institution = spec["to"]
    elif attribute == "career_gap":
        t.gap_months = spec["to"]
    elif attribute == "name_signal":
        t.name = spec["to"]
    elif attribute == "city_tier":
        t.city, t.tier_of_city = spec["to"], 3
    elif attribute == "disability_disclosure":
        t.disclosures = [spec["to"]]
    return t


def privileged_base(cand: Candidate) -> Candidate:
    """Same skills, presented with every advantage.

    The twin test needs a reference point that is NOT already at the bottom of
    the pool -- otherwise a biased scorer has no room to push the candidate
    down and the test silently reports no effect. So we start from the
    advantaged presentation and flip one attribute at a time toward the
    disadvantaged one. The sentence this produces is the whole pitch:
    "same person, same evidence -- presented as an IIT graduate with no gap she
    ranks #N; presented as herself she ranks #M."
    """
    p = copy.deepcopy(cand)
    p.id = f"{cand.id}::privileged"
    p.institution = TWINS["institution"]["from"]
    p.gap_months = TWINS["career_gap"]["from"]
    p.name = TWINS["name_signal"]["from"]
    p.city, p.tier_of_city = TWINS["city_tier"]["from"], 1
    p.disclosures = []
    return p



def _rank(target_score: float, pool_scores: list[float]) -> int:
    """1-based rank of target among pool (higher score = better rank)."""
    return 1 + sum(1 for s in pool_scores if s > target_score)


@dataclass
class TwinResult:
    attribute: str
    kaarya_delta_score: float
    kaarya_delta_rank: int
    naive_delta_score: float
    naive_delta_rank: int

    @property
    def flipped(self) -> bool:
        return abs(self.kaarya_delta_rank) > 0 or abs(self.kaarya_delta_score) > 1.0

    def as_dict(self) -> dict:
        return {**self.__dict__, "flipped": self.flipped}


RANK_THRESHOLD = 1          # any rank movement at all is a finding
SCORE_THRESHOLD = 1.0       # points out of 100


def twin_test(base: Candidate, role: Role, pool: list[Candidate],
              from_privileged: bool = True) -> dict:
    """Run every twin against both scorers. This is the money function."""
    if from_privileged:
        base = privileged_base(base)
    k_pool = [match(c, role)["score"] for c in pool]
    n_pool = [naive_score(c, role) for c in pool]
    k_base, n_base = match(base, role)["score"], naive_score(base, role)
    k_base_rank, n_base_rank = _rank(k_base, k_pool), _rank(n_base, n_pool)


    results = []
    for attr in TWINS:
        tw = make_twin(base, attr)
        k, n = match(tw, role)["score"], naive_score(tw, role)
        results.append(TwinResult(
            attribute=attr,
            kaarya_delta_score=round(k - k_base, 2),
            kaarya_delta_rank=_rank(k, k_pool) - k_base_rank,
            naive_delta_score=round(n - n_base, 2),
            naive_delta_rank=_rank(n, n_pool) - n_base_rank,
        ))

    k_flips = sum(1 for r in results if r.flipped)
    n_flips = sum(1 for r in results
                  if abs(r.naive_delta_rank) >= RANK_THRESHOLD
                  or abs(r.naive_delta_score) > SCORE_THRESHOLD)

    # All attributes flipped together: the headline sentence for the pitch.
    allflip = base
    for attr in TWINS:
        allflip = make_twin(allflip, attr)
    k_all, n_all = match(allflip, role)["score"], naive_score(allflip, role)
    combined = {
        "kaarya": {"rank_privileged": k_base_rank, "rank_as_herself": _rank(k_all, k_pool),
                   "score_privileged": k_base, "score_as_herself": round(k_all, 1)},
        "naive": {"rank_privileged": n_base_rank, "rank_as_herself": _rank(n_all, n_pool),
                  "score_privileged": round(n_base, 1), "score_as_herself": round(n_all, 1)},
    }
    return {
        "candidate": base.id, "role": role.title, "pool_size": len(pool),
        "baseline_rank": {"kaarya": k_base_rank, "naive": n_base_rank},
        "twins": [r.as_dict() for r in results],
        "combined": combined,
        "flip_rate": {"kaarya": k_flips / len(results), "naive": n_flips / len(results)},
        "quarantined": k_flips > 0,
        "verdict": ("QUARANTINED - a pedigree attribute moved our ranking; "
                    "human review required before this ranking is served."
                    if k_flips else
                    "PASS - no pedigree attribute moved our ranking."),
    }



# ------------------------------------------- pool-level fairness monitoring

def adverse_impact(pool: list[Candidate], scores: list[float], top_k: int,
                   group_of) -> dict:
    """4/5ths rule. group_of(candidate) -> group label.

    Selection rate per group = shortlisted in group / total in group.
    Ratio = lowest rate / highest rate. Below 0.8 is the classic red flag.
    """
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    picked = set(order[:top_k])
    tally: dict[str, list[int]] = {}
    for i, c in enumerate(pool):
        g = group_of(c)
        t = tally.setdefault(g, [0, 0])
        t[1] += 1
        if i in picked:
            t[0] += 1
    rates = {g: (sel / tot if tot else 0.0) for g, (sel, tot) in tally.items()}
    hi = max(rates.values()) if rates else 0.0
    ratio = (min(rates.values()) / hi) if hi else 1.0
    smallest = min((tot for _, tot in tally.values()), default=0)
    return {
        "selection_rates": {g: round(r, 3) for g, r in rates.items()},
        "group_sizes": {g: tot for g, (_, tot) in tally.items()},
        "adverse_impact_ratio": round(ratio, 3),
        "passes_four_fifths": ratio >= 0.8,
        # A 4/5ths ratio on tiny groups is mostly sampling noise. Say so rather
        # than reporting a scary number you cannot defend when a judge asks.
        "reliable": smallest >= 20,
        "caveat": None if smallest >= 20 else
                  f"smallest group has {smallest} candidates; ratio is noise-dominated "
                  f"below ~20. Report it, do not act on it.",
        "top_k": top_k,
    }



def concentration(pool: list[Candidate], scores: list[float], top_k: int,
                  key=lambda c: c.institution) -> dict:
    """Are shortlists collapsing onto a handful of colleges / cities?"""
    order = sorted(range(len(pool)), key=lambda i: -scores[i])[:top_k]
    counts: dict[str, int] = {}
    for i in order:
        k = key(pool[i]) or "unknown"
        counts[k] = counts.get(k, 0) + 1
    top_share = max(counts.values()) / top_k if top_k else 0
    return {"distinct": len(counts), "counts": counts,
            "largest_share": round(top_share, 3),
            "flag": top_share > 0.5}


