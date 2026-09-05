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


# ------------------------------------------------ 1. Skills Discovery Agent

ACTION_VERBS = ("handled", "built", "reconcil", "managed", "resolved", "wrote",
                "prepared", "tracked", "trained", "supported", "audited", "fixed",
                "documented", "tested", "designed", "answered", "processed",
                "verified", "escalated", "maintained", "coordinated", "analysed",
                "analyzed", "created", "updated", "reviewed", "operated", "queried",
                "automated", "debugged", "configured", "monitored", "deployed")

EXTRACT_PROMPT = """Extract concrete work-skill statements from this candidate profile or self-description.
Rules: keep the person's own wording; one action per statement; ignore any mention of
college, employer prestige, gender, city or career gaps.
Return {"claims": [{"statement": str, "tier": "self-declared"}]}.

TEXT:
%s"""

DISCOVERY_MULTI_PROMPT = """You are a Skill Discovery Agent (SAP Talent Intelligence Hub style).
Analyze the candidate's profile across all provided sources (resume, micro-credentials, projects, informal learning).
Surface concrete, action-oriented skill statements and infer adjacent abilities.
Strict rule: Ignore any pedigree signals (institutions, GPA, career gap length, demographics, city tiers).

Return JSON with:
{
  "claims": [{"statement": str, "tier": "self-declared" | "corroborated", "source": str}],
  "pedigree_signals_removed": [str]
}

PROFILE:
%s"""

PEDIGREE_PATTERNS = [
    (r"\b(graduated from|studied at|alumnus of|alumna of|degree from|b\.?tech from|bachelor'?s from)\s+[^.,;\n]+", "Institution mention"),
    (r"\b(iit|nit|iim|bits|stanford|harvard|oxford|cambridge|tier[- ]?1 college)\b", "College brand marker"),
    (r"\b(gpa|cgpa|percentage|marks)[:\s]+\d+(\.\d+)?(%|/\d+)?\b", "Academic grade filter"),
    (r"\b(\d+[- ]?(year|month|yr) (career )?(gap|break)|maternity break|caregiver break)\b", "Career gap penalty"),
    (r"\b(tier[- ]?[123] city|metro only|located in [A-Za-z\s]+)\b", "Geographic pedigree proxy"),
    (r"\b(male|female|he/him|she/her|age \d+|\d+ years old)\b", "Demographic attribute"),
]


def strip_pedigree_signals(text: str) -> tuple[str, list[str]]:
    """Strip credential and pedigree markers before skill extraction."""
    removed = []
    cleaned = text
    for pat, label in PEDIGREE_PATTERNS:
        matches = re.findall(pat, cleaned, flags=re.I)
        if matches:
            for m in matches:
                matched_str = m[0] if isinstance(m, tuple) else m
                removed.append(f"{label}: '{matched_str.strip()}'")
            cleaned = re.sub(pat, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, removed


# ---------------------------------- Skill Knowledge Graph & Ontology (TIH Style)

SKILL_GRAPH_NODES = {
    "spreadsheet_reconciliation": {
        "title": "Spreadsheet Reconciliation & Ledger Audit",
        "domain": "Financial & Data Operations",
        "keywords": ["reconcil", "ledger", "invoice", "excel", "spreadsheet", "duplicate", "vlookup"],
        "canonical_statement": "Reconciled financial records, invoices, and ledgers in spreadsheets to resolve discrepancies",
        "adjacencies": [
            {"target": "sql_data_validation", "relation": "adjacent_to", "weight": 0.85,
             "explanation": "Cross-system ledger reconciliation frequently uses SQL query verification."},
            {"target": "data_cleaning_dedup", "relation": "frequently_cooccurs_with", "weight": 0.90,
             "explanation": "Duplicate record identification directly translates to structured data deduplication."},
            {"target": "financial_reporting", "relation": "specializes", "weight": 0.80,
             "explanation": "Reconciled ledger data feeds period-end balance sheet and variance reports."},
        ],
    },
    "sql_data_validation": {
        "title": "SQL Database Querying & Data Validation",
        "domain": "Data Operations",
        "keywords": ["sql", "query", "database", "rdbms", "table", "join", "select"],
        "canonical_statement": "Queried relational databases and validated data records across source tables",
        "adjacencies": [
            {"target": "spreadsheet_reconciliation", "relation": "adjacent_to", "weight": 0.85,
             "explanation": "Validating relational table outputs pairs with spreadsheet audit routines."},
            {"target": "etl_pipeline_monitoring", "relation": "adjacent_to", "weight": 0.78,
             "explanation": "Query validation skills underpin automated ETL data extraction checks."},
            {"target": "bi_dashboard_reporting", "relation": "prerequisite_for", "weight": 0.82,
             "explanation": "SQL extraction is foundational for feeding business intelligence metrics."},
        ],
    },
    "data_cleaning_dedup": {
        "title": "Data Cleaning & Anomaly Resolution",
        "domain": "Data Operations",
        "keywords": ["clean", "dedup", "duplicate", "null", "transform", "normalize", "sanitize"],
        "canonical_statement": "Cleaned datasets, eliminated duplicate entries, and corrected inconsistent records",
        "adjacencies": [
            {"target": "spreadsheet_reconciliation", "relation": "adjacent_to", "weight": 0.90,
             "explanation": "Core hygiene step in financial and operational ledger maintenance."},
            {"target": "sql_data_validation", "relation": "adjacent_to", "weight": 0.80,
             "explanation": "Automated deduplication logic is commonly authored in SQL transformations."},
        ],
    },
    "financial_reporting": {
        "title": "Financial Variance & Management Reporting",
        "domain": "Financial Operations",
        "keywords": ["balance sheet", "variance report", "financial report", "period end", "p&l", "budget"],
        "canonical_statement": "Prepared periodic financial reports and analyzed budget variance metrics",
        "adjacencies": [
            {"target": "spreadsheet_reconciliation", "relation": "prerequisite_for", "weight": 0.85,
             "explanation": "Accurate ledger reconciliation forms the factual foundation for management reports."},
            {"target": "bi_dashboard_reporting", "relation": "adjacent_to", "weight": 0.80,
             "explanation": "Financial metrics are visualized through operational reporting dashboards."},
        ],
    },
    "etl_pipeline_monitoring": {
        "title": "ETL Pipeline & Data Flow Monitoring",
        "domain": "Data Engineering",
        "keywords": ["etl", "pipeline", "ingestion", "data flow", "batch job", "cron"],
        "canonical_statement": "Monitored scheduled ETL data pipelines and verified data ingestion completeness",
        "adjacencies": [
            {"target": "sql_data_validation", "relation": "frequently_cooccurs_with", "weight": 0.82,
             "explanation": "Pipeline validation relies on automated SQL sanity checks."},
        ],
    },
    "bi_dashboard_reporting": {
        "title": "Business Intelligence (BI) Dashboarding",
        "domain": "Analytics & Reporting",
        "keywords": ["power bi", "tableau", "dashboard", "kpi", "visualization", "metrics"],
        "canonical_statement": "Built interactive business intelligence dashboards to track organizational KPIs",
        "adjacencies": [
            {"target": "sql_data_validation", "relation": "prerequisite_for", "weight": 0.85,
             "explanation": "Dashboards require structured SQL query models for reliable metric feeds."},
        ],
    },
    "crm_case_management": {
        "title": "CRM Case & Ticket Resolution",
        "domain": "Customer & Business Operations",
        "keywords": ["crm", "ticket", "case", "zendesk", "salesforce", "jira", "inbound", "queue"],
        "canonical_statement": "Managed and resolved customer support tickets and case logs in CRM software",
        "adjacencies": [
            {"target": "customer_deescalation", "relation": "frequently_cooccurs_with", "weight": 0.88,
             "explanation": "Handling complex CRM tickets directly involves customer de-escalation skills."},
            {"target": "sla_tracking", "relation": "adjacent_to", "weight": 0.82,
             "explanation": "Managing ticket queues requires tracking SLA response and resolution times."},
            {"target": "process_documentation", "relation": "prerequisite_for", "weight": 0.78,
             "explanation": "Documenting resolved issues forms standard troubleshooting knowledge bases."},
        ],
    },
    "customer_deescalation": {
        "title": "Customer Conflict De-escalation",
        "domain": "Customer Operations",
        "keywords": ["de-escalat", "escalat", "dispute", "complaint", "conflict", "retention"],
        "canonical_statement": "De-escalated customer disputes and mediated resolution for high-priority complaints",
        "adjacencies": [
            {"target": "crm_case_management", "relation": "frequently_cooccurs_with", "weight": 0.88,
             "explanation": "De-escalation outcomes are documented and tracked via CRM workflows."},
            {"target": "multi_channel_support", "relation": "adjacent_to", "weight": 0.84,
             "explanation": "De-escalation applies across live chat, email, and ticketing channels."},
        ],
    },
    "sla_tracking": {
        "title": "SLA Tracking & Queue Operations",
        "domain": "Operations Management",
        "keywords": ["sla", "service level", "turnaround time", "queue priority", "backlog"],
        "canonical_statement": "Monitored support queue response times to maintain SLA turnaround compliance",
        "adjacencies": [
            {"target": "crm_case_management", "relation": "frequently_cooccurs_with", "weight": 0.85,
             "explanation": "SLA metrics guide ticket prioritization in CRM systems."},
        ],
    },
    "multi_channel_support": {
        "title": "Omnichannel Customer Communication",
        "domain": "Customer Support",
        "keywords": ["chat support", "email support", "omnichannel", "messaging support", "helpdesk"],
        "canonical_statement": "Delivered asynchronous customer support across live chat, email, and ticketing portals",
        "adjacencies": [
            {"target": "customer_deescalation", "relation": "adjacent_to", "weight": 0.85,
             "explanation": "De-escalation techniques adapt to written asynchronous chat channels."},
        ],
    },
    "process_documentation": {
        "title": "Standard Operating Procedure (SOP) Authoring",
        "domain": "Operational Excellence",
        "keywords": ["document", "sop", "guide", "workflow", "manual", "handbook", "procedure"],
        "canonical_statement": "Authored standard operating procedures and technical documentation for team workflows",
        "adjacencies": [
            {"target": "crm_case_management", "relation": "adjacent_to", "weight": 0.78,
             "explanation": "Capturing case resolutions creates repeatable support playbooks."},
            {"target": "quality_assurance_audit", "relation": "prerequisite_for", "weight": 0.80,
             "explanation": "Clear SOPs enable systematic peer review and compliance auditing."},
        ],
    },
    "quality_assurance_audit": {
        "title": "Quality Assurance & Workflow Auditing",
        "domain": "Operational Excellence",
        "keywords": ["qa audit", "quality assurance", "compliance check", "peer review", "inspection"],
        "canonical_statement": "Audited operational workflows and ticket resolutions against QA compliance standards",
        "adjacencies": [
            {"target": "process_documentation", "relation": "adjacent_to", "weight": 0.82,
             "explanation": "QA audits evaluate adherence to established SOP documentation."},
        ],
    },
    "inventory_logistics_tracking": {
        "title": "Inventory & Shipment Tracking",
        "domain": "Logistics & Supply Chain",
        "keywords": ["inventory", "shipment", "warehouse", "stock", "dispatch", "order tracking"],
        "canonical_statement": "Tracked warehouse stock levels, purchase orders, and dispatched shipments",
        "adjacencies": [
            {"target": "vendor_order_management", "relation": "adjacent_to", "weight": 0.84,
             "explanation": "Inventory management directly interfaces with vendor purchase order tracking."},
            {"target": "spreadsheet_reconciliation", "relation": "adjacent_to", "weight": 0.76,
             "explanation": "Periodic stock counts are reconciled against procurement ledgers."},
        ],
    },
    "vendor_order_management": {
        "title": "Vendor Purchase Order Management",
        "domain": "Supply Chain & Procurement",
        "keywords": ["purchase order", "po tracking", "vendor coordination", "supplier delivery"],
        "canonical_statement": "Managed vendor purchase orders, delivery timelines, and fulfillment tracking",
        "adjacencies": [
            {"target": "inventory_logistics_tracking", "relation": "adjacent_to", "weight": 0.84,
             "explanation": "Purchase orders directly correspond to inbound inventory batches."},
        ],
    },
}


class SkillKnowledgeGraph:
    """Skill Knowledge Graph for adjacent skill discovery (Talent Intelligence Hub style)."""

    def __init__(self, nodes: dict = SKILL_GRAPH_NODES):
        self.nodes = nodes

    def match_node(self, text: str) -> tuple[str | None, float]:
        """Find the closest knowledge graph node for a given skill text."""
        s = text.lower()
        best_node, best_score = None, 0.0
        for node_id, data in self.nodes.items():
            matches = sum(1 for kw in data["keywords"] if kw in s)
            if matches > 0:
                score = min(1.0, matches * 0.35 + 0.30)
                if score > best_score:
                    best_node, best_score = node_id, score
        return best_node, best_score

    def infer_adjacent_skills(self, claims: list[SkillClaim],
                              top_k: int = 4, min_confidence: float = 0.40) -> list[dict]:
        """Traverse the knowledge graph to infer adjacent and prerequisite capabilities."""
        existing_statements = " ".join(c.statement.lower() for c in claims)
        inferred = []
        seen_targets = set()

        for claim in claims:
            matched_node_id, match_score = self.match_node(claim.statement)
            if not matched_node_id:
                continue
            node = self.nodes[matched_node_id]
            for adj in node["adjacencies"]:
                target_id = adj["target"]
                if target_id in seen_targets:
                    continue
                target_node = self.nodes.get(target_id)
                if not target_node:
                    continue

                # Only skip if the exact title or canonical statement is largely present
                if target_node["title"].lower() in existing_statements:
                    continue

                confidence = round(match_score * adj["weight"], 3)
                if confidence >= min_confidence:
                    seen_targets.add(target_id)
                    inferred.append({
                        "skill_id": target_id,
                        "title": target_node["title"],
                        "statement": target_node["canonical_statement"],
                        "confidence": confidence,
                        "relation": adj["relation"],
                        "domain": target_node["domain"],
                        "inferred_from": claim.statement,
                        "explanation": adj["explanation"],
                    })

        inferred.sort(key=lambda x: x["confidence"], reverse=True)
        return inferred[:top_k]


_SKILL_GRAPH = SkillKnowledgeGraph()


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
    """Skills Discovery: Extracts work-skill statements, stripping pedigree signals."""
    cleaned_text, _ = strip_pedigree_signals(text)
    out = llm(EXTRACT_PROMPT % cleaned_text)
    if out:
        try:
            return [SkillClaim(statement=c["statement"],
                               tier=c.get("tier", "self-declared"))
                    for c in json.loads(out)["claims"]][:12]
        except Exception:                        # noqa: BLE001
            pass
    claims = [s for s in _segment(cleaned_text)
              if len(s.split()) >= 3 and any(v in s.lower() for v in ACTION_VERBS)]
    if not claims:      # no verb we recognise: fall back to punctuation clauses
        claims = [s.strip() for s in re.split(r"[.;,\n]", cleaned_text)
                  if len(s.split()) >= 4]
    seen: set[str] = set()
    uniq = [s for s in claims if not (s.lower() in seen or seen.add(s.lower()))]
    return ([SkillClaim(statement=s) for s in uniq[:12]]
            or [SkillClaim(statement=cleaned_text.strip()[:200])])


def discover_skills(profile_data: str | dict,
                    include_adjacent: bool = True) -> dict:
    """Complete Skills Discovery Agent.

    Analyzes multi-source candidate profiles (resumes, projects, micro-credentials,
    informal learning, self-descriptions) and infers adjacent skills via the
    Skill Knowledge Graph. Crucially strips pedigree markers to surface pure ability.
    """
    direct_claims: list[SkillClaim] = []
    pedigree_removed: list[str] = []
    sources_analyzed: list[str] = []

    if isinstance(profile_data, str):
        cleaned, removed = strip_pedigree_signals(profile_data)
        pedigree_removed.extend(removed)
        sources_analyzed.append("unstructured_profile")
        direct_claims.extend(extract_claims(cleaned))
    elif isinstance(profile_data, dict):
        # 1. Self-description / Resume text
        if profile_data.get("text") or profile_data.get("resume_text"):
            raw = str(profile_data.get("text") or profile_data.get("resume_text"))
            cleaned, removed = strip_pedigree_signals(raw)
            pedigree_removed.extend(removed)
            sources_analyzed.append("resume_text")
            direct_claims.extend(extract_claims(cleaned))

        # 2. Micro-credentials / Certifications (corroborated)
        micro_creds = profile_data.get("micro_credentials") or profile_data.get("certifications") or []
        if isinstance(micro_creds, list):
            for mc in micro_creds:
                sources_analyzed.append("micro_credentials")
                if isinstance(mc, dict):
                    name = mc.get("name", "Certification")
                    skills = mc.get("skills", [])
                    issuer = mc.get("issuer", "verified_provider")
                    if skills:
                        for sk in skills:
                            direct_claims.append(SkillClaim(
                                statement=f"Demonstrated {sk} via {name}",
                                tier="corroborated", source=f"cert:{issuer}"))
                    else:
                        direct_claims.append(SkillClaim(
                            statement=f"Completed {name} covering core competencies",
                            tier="corroborated", source=f"cert:{issuer}"))
                elif isinstance(mc, str):
                    cleaned, _ = strip_pedigree_signals(mc)
                    direct_claims.append(SkillClaim(
                        statement=cleaned, tier="corroborated", source="micro_credential"))

        # 3. Project Work / Portfolios (corroborated with project artifacts)
        projects = profile_data.get("projects") or profile_data.get("portfolio") or []
        if isinstance(projects, list):
            for proj in projects:
                sources_analyzed.append("project_work")
                if isinstance(proj, dict):
                    title = proj.get("title", "Project")
                    desc = proj.get("description", "")
                    cleaned_desc, _ = strip_pedigree_signals(desc)
                    claims_from_proj = _segment(cleaned_desc)
                    if claims_from_proj:
                        for cl in claims_from_proj[:3]:
                            direct_claims.append(SkillClaim(
                                statement=f"{cl} on {title}",
                                tier="corroborated", source=f"project:{title}"))
                    elif cleaned_desc:
                        direct_claims.append(SkillClaim(
                            statement=f"{title}: {cleaned_desc}",
                            tier="corroborated", source=f"project:{title}"))
                elif isinstance(proj, str):
                    cleaned_proj, _ = strip_pedigree_signals(proj)
                    direct_claims.append(SkillClaim(
                        statement=cleaned_proj, tier="corroborated", source="project"))

        # 4. Informal Learning & Open Source
        informal = profile_data.get("informal_learning") or []
        if isinstance(informal, list):
            for item in informal:
                sources_analyzed.append("informal_learning")
                cleaned_item, _ = strip_pedigree_signals(str(item))
                direct_claims.append(SkillClaim(
                    statement=cleaned_item, tier="self-declared", source="informal_learning"))

        # 5. Explicit self-reported skills
        self_skills = profile_data.get("self_reported_skills") or profile_data.get("capabilities") or []
        if isinstance(self_skills, list):
            for sk in self_skills:
                sources_analyzed.append("self_reported_skills")
                cleaned_sk, _ = strip_pedigree_signals(str(sk))
                direct_claims.append(SkillClaim(
                    statement=cleaned_sk, tier="self-declared", source="self"))

    # Deduplicate direct claims
    seen_stmts: set[str] = set()
    deduped_claims: list[SkillClaim] = []
    for c in direct_claims:
        key = c.statement.lower().strip()
        if key and key not in seen_stmts:
            seen_stmts.add(key)
            deduped_claims.append(c)

    # Infer adjacent skills via the Skill Knowledge Graph
    adjacent_skills: list[dict] = []
    if include_adjacent and deduped_claims:
        adjacent_skills = _SKILL_GRAPH.infer_adjacent_skills(deduped_claims)

    inferred_claims = [
        SkillClaim(statement=adj["statement"], tier="inferred",
                   source=f"knowledge_graph:{adj['skill_id']}")
        for adj in adjacent_skills
    ]

    return {
        "claims": deduped_claims,
        "adjacent_skills": adjacent_skills,
        "sources_analyzed": list(dict.fromkeys(sources_analyzed)),
        "pedigree_filtered": pedigree_removed,
        "all_claims": deduped_claims + inferred_claims,
    }


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



