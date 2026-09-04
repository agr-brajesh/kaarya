"""End-to-end smoke test AND the script your demo narrates. Run: python demo.py"""
from __future__ import annotations

import agents
import audit
import seed
from engine import embedder, load_role, match, naive_score


def line(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main():
    con = seed.build_db()
    print(f"embedder: {embedder().name}")
    support = load_role(con, "43-4051.00")
    dataops = load_role(con, "13-1111.00")
    pool = seed.make_pool(40)

    # ---- TRACE A: Meena, no resume, 26-month gap, tier-3 college -------
    line("TRACE A  Meena -> Data Operations Analyst")
    claims = agents.extract_claims(seed.MEENA_TEXT)
    print(f"Evidence Agent extracted {len(claims)} claims, all self-declared")
    m = match(seed.MEENA, dataops)
    print(f"  before micro-task: score={m['score']}  {m['counts']}")

    agents.promote(seed.MEENA.claims, "reconciled vendor invoices",
                   "demonstrated", "micro-task:invoice-clean")
    agents.promote(seed.MEENA.claims, "built spreadsheet summaries",
                   "demonstrated", "micro-task:invoice-clean")
    m = match(seed.MEENA, dataops)
    print(f"  after  micro-task: score={m['score']}  {m['counts']}")
    print("  learning plan is derived from the gaps only:")
    for t in m["tasks"]:
        if t["verdict"] in ("coachable", "gap"):
            print(f"    - [{t['verdict']:9}] {t['task'][:62]}")

    # ---- THE TWIN TEST -------------------------------------------------
    line("TWIN TEST  same skills, one pedigree attribute flipped")
    r = audit.twin_test(seed.MEENA, dataops, pool)
    print("start from the ADVANTAGED presentation, then flip one thing at a time")
    print(f"baseline rank  kaarya #{r['baseline_rank']['kaarya']}   "
          f"naive #{r['baseline_rank']['naive']}   (pool {r['pool_size']})")
    print(f"\n{'attribute flipped':24}{'KAARYA drank':>14}{'NAIVE drank':>13}"
          f"{'NAIVE dscore':>14}")
    for t in r["twins"]:
        print(f"{t['attribute']:24}{t['kaarya_delta_rank']:>14}"
              f"{t['naive_delta_rank']:>13}{t['naive_delta_score']:>14}")
    c = r["combined"]
    print(f"\nALL FLIPPED (same person, presented as herself):")
    print(f"  kaarya  #{c['kaarya']['rank_privileged']} -> #{c['kaarya']['rank_as_herself']}"
          f"   score {c['kaarya']['score_privileged']} -> {c['kaarya']['score_as_herself']}")
    print(f"  naive   #{c['naive']['rank_privileged']} -> #{c['naive']['rank_as_herself']}"
          f"   score {c['naive']['score_privileged']} -> {c['naive']['score_as_herself']}")
    print(f"\nflip rate   kaarya {r['flip_rate']['kaarya']:.0%}   "
          f"naive {r['flip_rate']['naive']:.0%}")
    print(r["verdict"])

    # ---- POOL FAIRNESS -------------------------------------------------
    line("POOL FAIRNESS  top-20 shortlist")
    big = seed.make_pool(200, seed=11)
    k = [match(c, dataops)["score"] for c in big]
    n = [naive_score(c, dataops) for c in big]
    for label, scores in (("kaarya", k), ("naive ", n)):
        ai = audit.adverse_impact(big, scores, 20,
                                  lambda c: "gap" if c.gap_months else "no-gap")
        cc = audit.concentration(big, scores, 20)
        print(f"{label}  adverse-impact ratio={ai['adverse_impact_ratio']:.2f} "
              f"pass4/5={ai['passes_four_fifths']!s:5} reliable={ai['reliable']!s:5} "
              f"colleges in top-20={cc['distinct']} largest_share={cc['largest_share']:.0%}")


    # ---- TRACE B: Arjun, deaf, support role ----------------------------
    line("TRACE B  Arjun (deaf) -> Customer Support Specialist")
    m = match(seed.ARJUN, support)
    print(f"score={m['score']}  accessible={m['accessible_pct']}%  of {m['n_tasks']} tasks")
    print(f"  counts: {m['counts']}")
    print(f"  accommodations that resolve: {m['accommodations_needed']}")
    print(f"  genuinely unresolved: {m['unresolved_tasks']}")
    c = agents.carve(m)
    print(f"\nCarving needed={c['needed']}  carved weight={c['carved_weight']:.1%}")
    for p in c["proposals"]:
        tag = "RECOMMEND" if p["recommended"] else "tradeoff "
        print(f"  [{tag}] {p['type']:8} coverage_after={p['coverage_after']}%  {p['detail']}")

    # ---- EMPLOYER READINESS -------------------------------------------
    line("REQUISITION AGENT  job description scan")
    f = agents.scan_jd(support.jd_text)
    for x in f:
        print(f"  '{x['phrase']}' -> {x['action']}: {x['reason']}")
    print(f"\nrewritten JD:\n  {agents.rewrite_jd(support.jd_text, f)}")

    reqs = ["Bachelor's degree in engineering",
            "Must be able to handle high call volume",
            "Experience updating CRM records after customer contact"]
    print("\nphantom requirement check:")
    for p in agents.phantom_requirements(reqs, support):
        print(f"  phantom={p['phantom']!s:5} sim={p['best_task_similarity']:.2f}  "
              f"{p['requirement']}")


if __name__ == "__main__":
    main()
