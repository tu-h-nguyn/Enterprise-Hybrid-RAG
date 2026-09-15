#!/usr/bin/env python3
"""Generate in-domain distractor documents for the corpus-scale experiment.

    python scripts/make_distractor_corpus.py --count 2500 --out /tmp/distractors

Why these documents exist
-------------------------
The evaluation corpus is 22 documents / 77 chunks. At that size ``Recall@5``
saturates — five chunks is more than a tenth of the whole corpus — so the
headline table cannot answer the obvious question: *does the hybrid + reranker
advantage survive when the gold chunk is one in five thousand?*

``scripts/scale_experiment.py`` answers it by holding the 50 labelled questions
and their answer spans fixed and growing the haystack with the documents this
script writes. Only the corpus size changes, so any movement in Recall@1 or MRR
is attributable to corpus size and nothing else.

What makes a *useful* distractor
--------------------------------
Padding the corpus with off-topic text (novels, news wire) would prove nothing:
a policy question never retrieves a novel, so recall would stay flat and the
experiment would be a rigged win. These documents are therefore written in the
**same register and on the same topics** as the real corpus — leave policy,
incident response, vendor risk, retention schedules — for other fictional
companies, with their own entities, reference codes and numbers.

That makes them genuine hard negatives: they share BM25 vocabulary with the
questions ("annual leave", "recovery time objective", "must be approved by")
and they land in the same neighbourhood of embedding space. ``scale_experiment``
reports how often a distractor actually outranks the gold chunk, so the claim
that they are hard is measured rather than asserted.

Two invariants this script must preserve, both asserted downstream:

*   **No answer span may appear in a distractor.** Relevance is resolved per
    source document (see ``app.evaluation.dataset``), so a distractor can never
    be *labelled* relevant — but if one restated a gold fact it would be a
    correct answer scored as a miss. ``scale_experiment`` refuses to run if any
    distractor chunk contains a labelled span.
*   **Deterministic output.** Everything derives from ``--seed``, so the same
    seed reproduces the same corpus byte for byte and the numbers in
    ``scale_results.json`` can be regenerated.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------- vocabulary
#: Fictional companies, none of which is the demo corpus's "Northwind Analytics".
COMPANY_HEADS = [
    "Arclight", "Baymont", "Cindervale", "Drakeholm", "Eastmere", "Fernwick",
    "Greyloch", "Harrowgate", "Ironvale", "Juniper Bay", "Kelbridge", "Larkspur",
    "Meridian Hollow", "Northcott", "Oakhurst", "Pellworth", "Quarrymoor",
    "Ravensmere", "Stonebrook", "Thornfield", "Underhill", "Vantage Point",
    "Westmarch", "Yarrowden", "Zephyr Reach",
]
COMPANY_TAILS = ["Systems", "Logistics", "Diagnostics", "Foundry", "Instruments",
                 "Partners", "Technologies", "Holdings"]

CITIES = ["Lisbon", "Tallinn", "Kraków", "Bergen", "Valencia", "Leeds", "Turin",
          "Ghent", "Aarhus", "Porto", "Graz", "Malmö"]
ROLES = ["Head of Operations", "Group Controller", "Director of Engineering",
         "Chief Information Officer", "Regional Manager", "Head of Legal",
         "Quality Lead", "Programme Director", "Head of People", "Site Manager"]
TEAMS = ["the Operations Board", "the Governance Committee", "the Platform Guild",
         "the Risk Forum", "the Service Management team", "the Change Advisory Board",
         "the Data Office", "the Controls Group"]


class TopicPack:
    """The topic-specific nouns a generic policy sentence needs to sound real."""

    def __init__(self, slug: str, title: str, code: str, sections: list[str],
                 artifacts: list[str], objects: list[str], actions: list[str],
                 metrics: list[tuple[str, str]]) -> None:
        self.slug = slug
        self.title = title
        self.code = code
        self.sections = sections
        self.artifacts = artifacts
        self.objects = objects
        self.actions = actions
        self.metrics = metrics  # (metric name, unit)


#: One pack per topic in the real corpus. Sharing the real corpus's topics is
#: the whole point — a distractor about a different subject is not a distractor.
TOPICS: list[TopicPack] = [
    TopicPack(
        "leave_policy", "Leave and Absence Policy", "LAP",
        ["Scope", "Annual Entitlement", "Requesting Leave", "Sickness Absence",
         "Carry Over", "Unpaid Leave", "Records"],
        ["a leave request", "the absence register", "a return-to-work note",
         "the entitlement statement"],
        ["annual leave", "compassionate leave", "study leave", "time off in lieu",
         "sickness absence"],
        ["approve a leave request", "record an absence", "extend a leave block",
         "convert leave into time off in lieu"],
        [("notice period", "working days"), ("approval turnaround", "working days"),
         ("carry-over cap", "days"), ("certification threshold", "consecutive days")],
    ),
    TopicPack(
        "information_security", "Information Security Standard", "ISS",
        ["Purpose", "Access Control", "Authentication", "Endpoint Protection",
         "Encryption", "Logging and Monitoring", "Exceptions"],
        ["an access request", "the asset register", "a security exception",
         "the key inventory"],
        ["privileged accounts", "service accounts", "workstation disks",
         "administrative interfaces", "shared credentials"],
        ["grant privileged access", "rotate a credential", "register an exception",
         "revoke a dormant account"],
        [("password minimum length", "characters"), ("session timeout", "minutes"),
         ("credential rotation interval", "days"), ("review cadence", "months")],
    ),
    TopicPack(
        "incident_response", "Incident Response Procedure", "IRP",
        ["Severity Levels", "Declaring an Incident", "Roles", "Communications",
         "Mitigation", "Post-Incident Review", "Evidence Handling"],
        ["an incident record", "the incident timeline", "a status page update",
         "the post-incident report"],
        ["a severity one incident", "a customer-visible outage", "a degraded dependency",
         "a security incident"],
        ["declare an incident", "page the on-call engineer", "appoint an incident commander",
         "close an incident record"],
        [("acknowledgement target", "minutes"), ("first update interval", "minutes"),
         ("review deadline", "working days"), ("escalation threshold", "minutes")],
    ),
    TopicPack(
        "vendor_risk", "Vendor Risk Assessment Procedure", "VRA",
        ["Assessment Tiers", "Due Diligence", "Contractual Controls", "Subprocessors",
         "Ongoing Monitoring", "Exit Planning"],
        ["a vendor questionnaire", "the risk register", "an assurance report",
         "the subprocessor list"],
        ["a tier one vendor", "a critical supplier", "a hosting provider",
         "a payroll processor"],
        ["onboard a new vendor", "reassess a supplier", "terminate a contract",
         "approve a subprocessor"],
        [("reassessment interval", "months"), ("notice of change", "days"),
         ("remediation window", "working days"), ("assurance validity", "months")],
    ),
    TopicPack(
        "data_retention", "Records Retention Schedule", "RRS",
        ["Purpose", "Retention Periods", "Legal Hold", "Deletion", "Backups",
         "Exceptions", "Review"],
        ["a deletion certificate", "the retention matrix", "a legal hold notice",
         "the disposal log"],
        ["payroll records", "application logs", "contract documents",
         "customer support transcripts", "recruitment records"],
        ["place a record on legal hold", "authorise a deletion run",
         "extend a retention period", "archive a dataset"],
        [("retention period", "years"), ("log retention", "days"),
         ("deletion window", "working days"), ("schedule review", "months")],
    ),
    TopicPack(
        "support_sla", "Customer Support Service Levels", "CSL",
        ["Coverage", "Priority Definitions", "Response Targets", "Escalation",
         "Service Credits", "Reporting"],
        ["a support ticket", "the escalation matrix", "a service credit claim",
         "the monthly service report"],
        ["a priority one ticket", "a billing query", "a feature request",
         "a data export request"],
        ["escalate a ticket", "claim a service credit", "reopen a closed ticket",
         "reclassify a ticket priority"],
        [("first response target", "minutes"), ("resolution target", "hours"),
         ("availability commitment", "percent"), ("credit claim window", "days")],
    ),
    TopicPack(
        "release_process", "Product Release Process", "PRP",
        ["Release Trains", "Entry Criteria", "Change Approval", "Rollout",
         "Rollback", "Release Notes", "Freeze Periods"],
        ["a release candidate", "the change record", "a rollback plan",
         "the release checklist"],
        ["a minor release", "a hotfix", "a schema migration", "a configuration change"],
        ["approve a release", "start a staged rollout", "trigger a rollback",
         "declare a code freeze"],
        [("soak period", "hours"), ("canary share", "percent"),
         ("approval window", "working days"), ("rollback target", "minutes")],
    ),
    TopicPack(
        "onboarding", "Engineering Onboarding Guide", "EOG",
        ["Before Day One", "First Week", "Access and Tooling", "Buddy System",
         "First Contribution", "Probation Review"],
        ["an onboarding checklist", "the access request form", "a buddy assignment",
         "the probation report"],
        ["a new joiner", "an internal transfer", "a returning employee",
         "a contractor placement"],
        ["provision a laptop", "assign an onboarding buddy", "complete the security module",
         "schedule the probation review"],
        [("first contribution target", "working days"), ("probation length", "months"),
         ("access provisioning time", "working days"), ("check-in cadence", "weeks")],
    ),
    TopicPack(
        "procurement", "Procurement and Purchasing Policy", "PPP",
        ["Authority Limits", "Purchase Orders", "Competitive Quotes", "Invoicing",
         "Payment Terms", "Contract Renewal"],
        ["a purchase order", "the supplier master record", "a signed contract",
         "the approval matrix"],
        ["software subscriptions", "professional services", "hardware purchases",
         "marketing spend"],
        ["raise a purchase order", "approve an invoice", "renew a contract",
         "add a supplier to the master record"],
        [("purchase order threshold", "US dollars"), ("payment terms", "days"),
         ("quote requirement", "quotations"), ("renewal notice", "days")],
    ),
    TopicPack(
        "model_governance", "Model Governance Policy", "MGP",
        ["Model Inventory", "Risk Tiering", "Validation", "Approval to Deploy",
         "Monitoring", "Retirement", "Documentation"],
        ["a model card", "the model inventory", "a validation report",
         "the monitoring dashboard"],
        ["a high-risk model", "a scoring model", "a forecasting model",
         "a third-party model"],
        ["register a model", "approve a model for production", "revalidate a model",
         "retire a model"],
        [("revalidation interval", "months"), ("drift alert threshold", "percent"),
         ("documentation deadline", "working days"), ("review cadence", "months")],
    ),
    TopicPack(
        "facilities", "Office and Facilities Guide", "OFG",
        ["Building Access", "Desk Booking", "Visitors", "Health and Safety",
         "Deliveries", "Out of Hours"],
        ["an access badge", "the desk booking system", "a visitor log",
         "the evacuation plan"],
        ["meeting rooms", "the loading bay", "the server room", "bicycle storage"],
        ["book a desk", "register a visitor", "request out-of-hours access",
         "report a facilities fault"],
        [("booking window", "working days"), ("badge validity", "months"),
         ("fault response", "hours"), ("drill frequency", "months")],
    ),
    TopicPack(
        "contractor", "Contractor Engagement Policy", "CEP",
        ["Engagement Criteria", "Approval", "Rates and Invoicing", "Access",
         "Extension", "Offboarding"],
        ["a statement of work", "the contractor register", "a timesheet",
         "the offboarding checklist"],
        ["a fixed-term contractor", "an agency placement", "a consultancy engagement",
         "an offshore supplier"],
        ["approve a statement of work", "extend an engagement", "submit a timesheet",
         "revoke contractor access"],
        [("engagement cap", "months"), ("timesheet deadline", "working days"),
         ("notice period", "days"), ("rate review", "months")],
    ),
    TopicPack(
        "data_processing", "Data Processing Agreement Standard", "DPA",
        ["Roles of the Parties", "Processing Instructions", "Security Measures",
         "Subprocessing", "Data Subject Requests", "Breach Notification",
         "Return and Deletion"],
        ["a processing record", "the subprocessor register", "a breach notification",
         "the transfer impact assessment"],
        ["personal data", "special category data", "pseudonymised data",
         "cross-border transfers"],
        ["appoint a subprocessor", "respond to a data subject request",
         "notify a personal data breach", "delete personal data on termination"],
        [("breach notification window", "hours"), ("request response time", "days"),
         ("deletion deadline", "days"), ("audit notice", "working days")],
    ),
    TopicPack(
        "ml_platform", "Machine Learning Platform Standards", "MPS",
        ["Environments", "Feature Store", "Training Jobs", "Artefact Registry",
         "Serving", "Cost Controls"],
        ["a training job specification", "the artefact registry", "a feature definition",
         "the serving manifest"],
        ["batch training jobs", "online inference endpoints", "feature pipelines",
         "GPU node pools"],
        ["promote an artefact to production", "register a feature",
         "request a GPU quota increase", "decommission an endpoint"],
        [("job timeout", "hours"), ("artefact retention", "days"),
         ("quota review", "months"), ("latency budget", "milliseconds")],
    ),
    TopicPack(
        "data_engineering", "Data Engineering Standards", "DES",
        ["Layered Architecture", "Naming", "Schema Changes", "Data Quality",
         "Orchestration", "Ownership"],
        ["a pipeline definition", "the data catalogue", "a schema change request",
         "the quality scorecard"],
        ["ingestion pipelines", "curated tables", "reference datasets",
         "downstream extracts"],
        ["publish a table to the curated layer", "deprecate a dataset",
         "register a data owner", "raise a schema change request"],
        [("freshness target", "hours"), ("deprecation notice", "days"),
         ("quality threshold", "percent"), ("review cadence", "months")],
    ),
    TopicPack(
        "recruitment", "Recruitment and Selection Process", "RSP",
        ["Requisitions", "Sourcing", "Interview Stages", "Assessment",
         "Offers", "Referrals", "Record Keeping"],
        ["a requisition", "the scorecard", "an offer letter", "the interview panel"],
        ["an internal candidate", "an agency candidate", "a referral",
         "a graduate applicant"],
        ["open a requisition", "schedule an interview panel", "extend an offer",
         "pay a referral bonus"],
        [("time to first response", "working days"), ("offer validity", "days"),
         ("referral bonus", "US dollars"), ("panel size", "interviewers")],
    ),
    TopicPack(
        "conduct", "Code of Conduct", "COC",
        ["Our Commitments", "Respectful Behaviour", "Conflicts of Interest",
         "Gifts and Hospitality", "Raising a Concern", "Investigations"],
        ["a declared conflict", "the gift register", "a concern report",
         "the investigation summary"],
        ["outside employment", "supplier hospitality", "political donations",
         "personal relationships at work"],
        ["declare a conflict of interest", "register a gift", "raise a concern",
         "appoint an investigator"],
        [("declaration deadline", "working days"), ("gift threshold", "US dollars"),
         ("acknowledgement target", "working days"), ("investigation target", "working days")],
    ),
    TopicPack(
        "compensation", "Compensation and Benefits Summary", "CBS",
        ["Pay Structure", "Review Cycle", "Bonus", "Pension", "Insurance",
         "Allowances", "Equity"],
        ["a salary review record", "the benefits portal", "a bonus statement",
         "the pension schedule"],
        ["base salary", "the annual bonus", "the pension contribution",
         "the wellbeing allowance"],
        ["run the annual salary review", "change a pension contribution",
         "claim a wellbeing allowance", "enrol in the insurance scheme"],
        [("employer contribution", "percent"), ("bonus target", "percent"),
         ("review cycle", "months"), ("allowance", "US dollars")],
    ),
    TopicPack(
        "continuity", "Business Continuity Plan", "BCP",
        ["Recovery Objectives", "Backup Regime", "Failover", "Crisis Team",
         "Testing", "Communications"],
        ["a recovery runbook", "the continuity register", "a test report",
         "the call tree"],
        ["the transactional database", "object storage", "the corporate intranet",
         "the payments gateway"],
        ["invoke the continuity plan", "run a failover test", "restore from backup",
         "convene the crisis team"],
        [("recovery time objective", "hours"), ("recovery point objective", "minutes"),
         ("backup retention", "days"), ("test frequency", "months")],
    ),
    TopicPack(
        "service_desk", "IT Service Desk Guide", "ISD",
        ["Contacting the Desk", "Ticket Categories", "Hardware Requests",
         "Software Requests", "Account Problems", "Loan Equipment"],
        ["a service desk ticket", "the equipment loan form", "a software request",
         "the knowledge base article"],
        ["laptop replacements", "monitor requests", "licence assignments",
         "password resets"],
        ["raise a service desk ticket", "borrow loan equipment",
         "request a software licence", "reset a locked account"],
        [("triage target", "minutes"), ("hardware lead time", "working days"),
         ("loan period", "working days"), ("licence review", "months")],
    ),
    TopicPack(
        "travel", "Travel and Expense Policy", "TEP",
        ["Booking", "Air Travel", "Accommodation", "Ground Transport",
         "Per Diem", "Claims", "Non-Reimbursable Items"],
        ["an expense claim", "the travel booking tool", "a receipt",
         "the approval trail"],
        ["domestic travel", "international travel", "client entertainment",
         "conference attendance"],
        ["book a flight", "submit an expense claim", "approve client entertainment",
         "claim mileage"],
        [("claim deadline", "days"), ("hotel cap", "US dollars"),
         ("per diem", "US dollars"), ("advance booking", "working days")],
    ),
    TopicPack(
        "remote_work", "Remote and Hybrid Working Policy", "RHW",
        ["Eligibility", "Working Pattern", "Equipment", "Working Abroad",
         "Health and Safety", "Review"],
        ["a remote working agreement", "the equipment register",
         "a workstation assessment", "the travel notification"],
        ["hybrid working", "fully remote arrangements", "temporary relocation",
         "cross-border working"],
        ["approve a remote working agreement", "claim an equipment stipend",
         "notify a period of working abroad", "complete a workstation assessment"],
        [("office attendance", "days per week"), ("equipment stipend", "US dollars"),
         ("notification threshold", "days"), ("agreement review", "months")],
    ),
]

# ------------------------------------------------------------ sentence bank
#: Generic policy "moves". Each is filled from the topic pack above, so the
#: prose stays formulaic in exactly the way real policy documents are.
#:
#: Number slots are *typed* (``{n_days}``, ``{n_months}``, ...) rather than a
#: bare ``{n}``: an untyped slot produces "reviewed every 72 months", which
#: reads as noise and would make the distractors easy to spot.
SENTENCES = [
    "{Object} is governed by this document and by the {code} control set.",
    "Requests to {action} must be submitted at least {n_days} working days in advance.",
    "{Artifact} must be approved by the {role} before it takes effect.",
    "The {metric} for {object} is {n_metric} {unit}.",
    "Where the {metric} cannot be met, {team} must be informed within {n_hours} hours.",
    "{Artifact} is retained for {n_months} months and is available to {team} on request.",
    "Employees based in {city} follow this document together with local law.",
    "{Team} reviews this document every {n_months} months; the current revision is {code}-{rev}.",
    "Any request to {action} is logged against {artifact} and reviewed by the {role}.",
    "Exceptions are granted for a maximum of {n_days} days and must be re-approved thereafter.",
    "{Object} may not be changed without written agreement from the {role}.",
    "A request to {action} that is not answered within {n_days} working days is escalated "
    "to {team}.",
    "The {metric} applies to {object} and is measured monthly.",
    "{Artifact} must record the requester, the approver and the date of approval.",
    "Failure to follow this procedure is handled under the disciplinary policy.",
    "{Team} publishes a summary of {object} to the intranet every {n_months} months.",
    "Training on {object} is mandatory for all staff in {city} within {n_days} days of joining.",
    "The threshold above which the {role} must approve is {n_metric} {unit}.",
    "{Object} is reported to {team} at the end of each quarter.",
    "Where this document conflicts with a customer contract, the contract prevails.",
    "Records supporting {artifact} are kept for {n_years} years after the engagement ends.",
    "The {role} may delegate approval of {artifact} to a named deputy for up to "
    "{n_days} days.",
    "Requests relating to {object} are handled in the order they are received.",
    "An audit of {object} is carried out every {n_months} months by {team}.",
    "{Team} maintains {artifact} and corrects errors within {n_days} working days of "
    "being notified.",
    "Staff who {action} without approval must report the fact to the {role} the same day.",
    "The {metric} is reviewed annually and may be changed only by {team}.",
    "{Artifact} is written in English and stored in the document management system.",
]


def _cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _number_for(unit: str, rng: random.Random) -> int:
    """Plausible magnitudes — a 4000-minute response target reads as noise."""
    if unit in {"minutes", "milliseconds"}:
        return rng.choice([5, 10, 15, 20, 30, 45, 60, 90, 120])
    if unit == "hours":
        return rng.choice([2, 4, 6, 8, 12, 24, 48, 72])
    if unit in {"days", "working days", "consecutive days"}:
        return rng.choice([1, 2, 3, 5, 7, 10, 14, 20, 30, 45, 60, 90])
    if unit == "months":
        return rng.choice([3, 6, 9, 12, 18, 24, 36])
    if unit == "years":
        return rng.choice([2, 3, 5, 6, 7, 10])
    if unit == "percent":
        return rng.choice([1, 2, 5, 8, 10, 15, 20, 25, 50, 75, 95, 99])
    if unit == "US dollars":
        return rng.choice([50, 100, 150, 250, 400, 500, 750, 1000, 2500, 5000])
    if unit == "days per week":
        return rng.choice([1, 2, 3, 4])
    return rng.choice([2, 3, 4, 5, 6, 8, 10, 12])


def _render_sentence(template: str, pack: TopicPack, code: str,
                     rng: random.Random) -> str:
    metric, unit = rng.choice(pack.metrics)
    slots: dict[str, object] = {
        "object": rng.choice(pack.objects),
        "artifact": rng.choice(pack.artifacts),
        "action": rng.choice(pack.actions),
        "metric": metric,
        "unit": unit,
        "role": rng.choice(ROLES),
        "team": rng.choice(TEAMS),
        "city": rng.choice(CITIES),
        "code": code,
        "rev": f"{rng.randint(1, 9)}.{rng.randint(0, 9)}",
        "n_days": _number_for("days", rng),
        "n_months": _number_for("months", rng),
        "n_hours": _number_for("hours", rng),
        "n_years": _number_for("years", rng),
        "n_metric": _number_for(unit, rng),
    }
    slots.update({key.capitalize(): _cap(value) for key, value in slots.items()
                  if isinstance(value, str)})
    return template.format(**slots)


def render_document(company: str, pack: TopicPack, rng: random.Random) -> str:
    """One document: a title, a reference code and four to six ordered sections.

    Sentence templates are drawn without replacement across the whole document.
    Sampling with replacement produced documents that repeated the same clause
    three times, which is both unrealistic and an easy tell for the retriever.
    """
    code = f"{pack.code}-{rng.randint(1, 9)}"
    n_sections = rng.randint(4, min(6, len(pack.sections)))
    start = rng.randint(0, len(pack.sections) - n_sections)
    sections = pack.sections[start:start + n_sections]

    sentences_needed = sum(rng.randint(3, 5) for _ in sections)
    pool = rng.sample(SENTENCES, k=min(sentences_needed, len(SENTENCES)))
    while len(pool) < sentences_needed:  # only for unusually long documents
        pool.extend(rng.sample(SENTENCES, k=min(sentences_needed - len(pool),
                                                len(SENTENCES))))

    lines = [f"# {company} {pack.title}", ""]
    lines.append(
        f"This document sets out how {company} manages {rng.choice(pack.objects)}. "
        f"It is reference {code} and applies to all {company} entities."
    )
    lines.append("")
    cursor = 0
    for section in sections:
        lines.append(f"## {section}")
        lines.append("")
        for _ in range(rng.randint(3, 5)):
            if cursor >= len(pool):
                break
            lines.append(_render_sentence(pool[cursor], pack, code, rng))
            lines.append("")
            cursor += 1
    return "\n".join(lines).rstrip() + "\n"


def generate(out_dir: Path, count: int, seed: int) -> list[Path]:
    """Write ``count`` distractor documents; deterministic in ``seed``.

    Documents are emitted company-major so that any prefix of the returned list
    covers every topic — the scale experiment uses nested prefixes, and a prefix
    that happened to be all leave policies would not be a fair haystack.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    companies = [f"{head} {tail}" for head in COMPANY_HEADS for tail in COMPANY_TAILS]
    rng.shuffle(companies)

    written: list[Path] = []
    index = 0
    for company in companies:
        for pack in TOPICS:
            if index >= count:
                return written
            path = out_dir / f"{index:05d}_{pack.slug}.md"
            path.write_text(render_document(company, pack, rng), encoding="utf-8")
            written.append(path)
            index += 1
    if index < count:
        raise ValueError(
            f"Only {index} unique (company, topic) documents available; "
            f"asked for {count}. Add companies or topics rather than repeating."
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20250915)
    args = parser.parse_args()

    written = generate(args.out, args.count, args.seed)
    total_chars = sum(p.stat().st_size for p in written)
    print(f"Wrote {len(written)} distractor documents to {args.out}")
    print(f"Mean size: {total_chars / max(1, len(written)):.0f} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
