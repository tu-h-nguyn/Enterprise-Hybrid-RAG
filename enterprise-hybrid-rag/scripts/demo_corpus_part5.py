"""Additional demo documents (part 5): one PDF, one DOCX and two Markdown files.

Adding more PDF and DOCX material matters for the evaluation: it means a
meaningful share of the answers must survive real PDF text extraction and real
DOCX style parsing, rather than only clean Markdown.
"""

from __future__ import annotations

Section = tuple[str, list[str]]

COMPENSATION: list[Section] = [
    ("Purpose and Scope", [
        "This document sets out how Northwind Analytics determines pay, reviews salaries and "
        "administers benefits. It covers all permanent employees and is reference COMP-2.4, "
        "effective 1 April 2025.",
        "Compensation decisions are made by the line manager, moderated by a calibration panel "
        "and approved by the Head of People Operations. Individual salaries are confidential.",
    ]),
    ("Salary Bands", [
        "Every role is mapped to one of seven levels, from L1 for entry level to L7 for "
        "principal and director roles. Each level has a band with a minimum, a midpoint and a "
        "maximum.",
        "The midpoint of each band is set at the median of the market benchmark for the role "
        "and location, using survey data refreshed every twelve months.",
        "An offer above the midpoint of a band requires approval from the Head of People "
        "Operations. An offer above the band maximum is not permitted; the role must be "
        "re-levelled instead.",
    ]),
    ("Annual Review Cycle", [
        "The salary review cycle runs once per year. Proposals are submitted in February, "
        "calibrated in March, and any increase takes effect from 1 April.",
        "Employees who joined after 1 October are not included in the following April cycle "
        "and are reviewed in the subsequent year.",
        "Out-of-cycle increases are permitted only for a promotion, a role change, or a "
        "documented retention risk approved by a member of the executive team.",
    ]),
    ("Promotion", [
        "Promotion requires demonstrated performance at the next level for at least two "
        "consecutive review periods, evidenced in a written promotion case.",
        "Promotion cases are reviewed twice a year, in March and in September. A promotion "
        "carries a minimum salary increase of 8 percent, subject to the band maximum.",
    ]),
    ("Variable Pay", [
        "All employees participate in a company bonus scheme with a target of 10 percent of "
        "base salary. The scheme pays between 0 and 150 percent of target depending on company "
        "performance against the annual operating plan.",
        "The sales team participates in a commission plan instead of the company bonus scheme. "
        "Commission is paid monthly and is subject to a clawback if a customer cancels within "
        "90 days of contract signature.",
        "Bonus is paid in March for the preceding calendar year. Employees must be employed and "
        "not under notice on the payment date to receive a bonus.",
    ]),
    ("Pension and Insurance", [
        "The company contributes 6 percent of base salary to a pension scheme and matches "
        "employee contributions up to a further 3 percent.",
        "Life insurance is provided at four times base salary. Income protection pays 60 "
        "percent of base salary after a 13-week deferral period.",
        "Private medical cover is available to all employees after three months of service and "
        "may be extended to a partner and dependent children at the employee's cost.",
    ]),
    ("Referral and Recognition", [
        "A successful employee referral pays 2000 euro for engineering roles and 1200 euro for "
        "all other roles. The bonus is paid after the referred employee completes three months "
        "of service.",
        "Spot recognition awards of up to 500 euro may be granted by any director without "
        "further approval.",
    ]),
]

CODE_OF_CONDUCT: list[Section] = [
    ("Our Standards", [
        "Northwind Analytics expects every employee, contractor and board member to act with "
        "honesty, respect and professional integrity. This code is reference GOV-COC-3 and is "
        "acknowledged by every employee on joining and annually thereafter.",
        "Breaching this code may result in disciplinary action up to and including summary "
        "dismissal, and may be reported to regulators or law enforcement where required.",
    ]),
    ("Conflicts of Interest", [
        "Employees must declare any outside employment, directorship, or financial interest in "
        "a competitor, customer or supplier of the company.",
        "A declaration must be filed within 14 days of the interest arising. Declarations are "
        "reviewed by the Legal team and recorded in the conflicts register.",
        "An employee must not participate in a procurement decision involving a supplier in "
        "which they or a close family member hold an interest.",
    ]),
    ("Gifts and Hospitality", [
        "Gifts with a value below 50 euro may be accepted without prior approval but must still "
        "be recorded in the gifts register.",
        "Gifts valued between 50 and 250 euro require written approval from a director. Gifts "
        "above 250 euro must be declined or donated to charity.",
        "Cash, cash equivalents and gift cards must never be accepted, regardless of value.",
    ]),
    ("Anti-Bribery and Corruption", [
        "The company prohibits bribery in any form, including facilitation payments, in every "
        "jurisdiction in which it operates.",
        "Any payment to a government official, including expedited processing fees, requires "
        "prior written approval from the General Counsel.",
    ]),
    ("Confidential Information", [
        "Confidential information includes customer data, unpublished financial results, "
        "product plans, security findings and personnel information.",
        "Confidentiality obligations continue for three years after employment ends. Trade "
        "secrets remain protected indefinitely.",
    ]),
    ("Raising Concerns", [
        "Concerns may be raised with a line manager, with the People Operations team, or "
        "anonymously through the independent whistleblowing line operated by a third party.",
        "Reports made through the whistleblowing line are acknowledged within 3 working days "
        "and an initial assessment is completed within 10 working days.",
        "The company prohibits retaliation against anyone who raises a concern in good faith. "
        "Retaliation is itself a disciplinary offence.",
    ]),
    ("Investigations", [
        "Investigations are conducted by a person independent of the matter. The subject of an "
        "investigation is informed unless doing so would prejudice the investigation.",
        "Investigation outcomes are recorded in writing. Summary statistics, with no "
        "identifying detail, are reported to the Audit Committee each quarter.",
    ]),
]

ML_PLATFORM_MD = """# Machine Learning Platform Standards

This document defines how Northwind Analytics builds, tracks and ships machine learning
models. It complements the Model Governance Policy and is reference ML-PLAT-2.

## Experiment Tracking

Every training run must be logged to the experiment tracking service with the git commit
hash, the dataset version identifier, the full hyperparameter set and the resulting metrics.

A run that is not reproducible from its logged metadata is not admissible as evidence for a
model promotion decision.

Experiment artefacts are retained for 18 months. Artefacts belonging to a model that reached
production are retained for the life of the model plus 3 years.

## Feature Store

Features used in production must be registered in the feature store with an owner, a
description, a data type and a freshness expectation.

Training and serving must read features through the same interface. Any feature computed
differently at training and serving time is treated as a defect, not as an acceptable
approximation.

## Model Registry

Models are registered with a semantic version. The major version is incremented when the
input schema changes, the minor version when the architecture or training data changes, and
the patch version for retraining on the same pipeline.

A model may occupy one of four stages: development, staging, production, or archived.

## Promotion to Production

Promotion to production requires an offline evaluation report, a fairness assessment where
the model affects individuals, a documented rollback plan, and sign-off from both the model
owner and the Model Risk Committee.

No model may be promoted to production on the basis of a single evaluation split. At minimum,
a temporal holdout and a cross-validated estimate must both be reported.

## Monitoring

Production models are monitored for input drift, prediction drift and performance
degradation. Drift is evaluated daily on a rolling 7-day window.

An alert is raised when the population stability index for any monitored feature exceeds
0.25, or when the primary performance metric degrades by more than 10 percent relative to the
validation baseline.

A model whose performance degrades by more than 25 percent must be rolled back or disabled
within 24 hours.

## Retraining

Scheduled retraining runs monthly for demand models and quarterly for risk models. An
unscheduled retrain is triggered by a sustained drift alert lasting more than 7 days.

Every retrained model is compared against the incumbent on the same holdout before
promotion. A retrained model is promoted only if it is at least as good as the incumbent on
the primary metric.
"""

DATA_QUALITY_MD = """# Data Engineering and Pipeline Standards

This document describes the standards for data pipelines at Northwind Analytics. It is owned
by the Data Platform team, reference DATA-ENG-7.

## Pipeline Service Levels

Tier A pipelines feed customer-facing analytics and must complete by 06:00 Central European
Time each day. Tier B pipelines feed internal reporting and must complete by 10:00.

A Tier A pipeline that has not completed by its deadline raises a page to the data on-call
engineer. A Tier B breach raises a ticket only.

Pipeline availability is measured as the percentage of scheduled runs completing within the
deadline. The target is 99 percent for Tier A and 95 percent for Tier B, measured monthly.

## Data Quality Checks

Every pipeline must implement freshness, volume, schema and null-rate checks on its output
tables.

A volume check fails when the row count deviates from the trailing 14-day median by more than
30 percent. A null-rate check fails when nulls in a non-nullable column exceed 0.1 percent.

A failed quality check blocks publication of the affected table. Downstream consumers read
the last known good partition until the issue is resolved.

## Schema Changes

Additive schema changes, such as adding a nullable column, may be deployed without notice.

Breaking schema changes, such as removing or renaming a column or narrowing a type, require
30 days notice to registered consumers and a deprecation period during which both old and new
fields are published.

Every table must declare an owner and a registered set of consumers. An unowned table is
scheduled for deletion after 90 days.

## Backfills

A backfill affecting more than 30 days of history requires approval from the data platform
lead and must run in a separate compute pool to avoid delaying scheduled pipelines.

Backfills must be idempotent. A pipeline that cannot be safely re-run for a past partition is
considered defective.

## Naming and Layout

Tables follow the pattern layer_domain_entity, where layer is one of raw, staged, curated or
mart.

Raw data is immutable and retained for 400 days. Curated tables are rebuilt from raw and may
be dropped and recreated at any time.

## Access

Access to curated and mart tables is granted by group membership, never to individual users.
Access to raw tables containing personal data requires approval from the data protection
officer and is reviewed every 6 months.
"""
