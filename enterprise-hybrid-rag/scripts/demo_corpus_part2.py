"""Second half of the demo corpus, split out to keep each module readable.

Several of these documents are *deliberately confusable* with the first half:
the contractor policy restates leave and expense rules with different numbers,
the data retention schedule repeats the 400-day log figure from the security
policy, and the release process overlaps the onboarding guide's CI/CD section.
Without such near-duplicates a retrieval benchmark is trivially easy and tells
you nothing about dense versus sparse behaviour.
"""

from __future__ import annotations

Section = tuple[str, list[str]]

EXTRA_HANDBOOK: list[Section] = [
    ("Learning and Development", [
        "Every employee has an annual learning budget of 2000 US dollars. The budget covers "
        "courses, books, conference tickets and professional memberships. It does not roll "
        "over into the next year.",
        "Employees may take up to 5 paid learning days per calendar year. Learning days are "
        "requested in the same system as annual leave and are approved by the line manager.",
        "Professional certification fees are reimbursed in full on a first successful "
        "attempt. Repeat attempts are reimbursed at 50 percent.",
    ]),
    ("Equipment and IT Support", [
        "Company laptops are refreshed every 3 years. Employees may keep their previous "
        "laptop only after it has been wiped and formally transferred by the IT team.",
        "Employees who take part in an on-call rotation receive a mobile phone allowance of "
        "40 US dollars per month.",
        "Lost or stolen equipment must be reported to the IT service desk within 24 hours so "
        "that the device can be remotely wiped.",
    ]),
    ("Business Conduct", [
        "Employees may not accept a gift from a supplier or customer with a value above 75 "
        "US dollars. Gifts above this value must be declined or surrendered to People "
        "Operations.",
        "Any potential conflict of interest must be disclosed in writing within 14 days of "
        "the employee becoming aware of it.",
        "Northwind Analytics prohibits facilitation payments of any kind, in any "
        "jurisdiction, without exception.",
    ]),
    ("Grievance Procedure", [
        "An employee should first raise a concern informally with their line manager. If "
        "that is not appropriate, the concern may be raised with People Operations directly.",
        "A formal written grievance receives a response within 10 working days. The employee "
        "may appeal the outcome within 5 working days of receiving it.",
        "An employee may be accompanied by a colleague at any formal grievance meeting.",
    ]),
]

EXTRA_SECURITY: list[Section] = [
    ("Network Security", [
        "Remote access to internal systems requires the corporate VPN and multi-factor "
        "authentication. Split tunnelling is disabled on all managed devices.",
        "Production networks are segmented from corporate networks. Firewall change requests "
        "are reviewed weekly by the network security team and require a documented business "
        "justification.",
        "Inbound access to production databases from the internet is prohibited without "
        "exception.",
    ]),
    ("Endpoint Management", [
        "All company devices must be enrolled in mobile device management before first use "
        "and must run the approved endpoint detection and response agent.",
        "Operating system and application patches are applied within 7 days for critical "
        "vulnerabilities, 14 days for high severity and 30 days for medium severity.",
        "Devices that have not checked in for 30 days are automatically blocked from "
        "accessing corporate resources.",
    ]),
    ("Secure Development", [
        "Static application security testing runs on every pull request. A high or critical "
        "finding blocks the merge until it is resolved or formally risk accepted.",
        "Dependency vulnerability scanning runs daily against all repositories. Critical "
        "dependency findings must be remediated within 7 days.",
        "An external penetration test is commissioned annually and after any significant "
        "change to the authentication architecture.",
        "A threat model is required before the first production deployment of any new "
        "service that processes Confidential or Restricted data.",
    ]),
    ("Business Continuity and Backup", [
        "The recovery point objective for production data is 1 hour. The recovery time "
        "objective is 4 hours.",
        "Database backups run daily and are retained for 35 days. Backup restoration is "
        "tested quarterly and the result is recorded in the risk register.",
        "A full disaster recovery exercise is performed annually with participation from "
        "engineering, security and customer support.",
    ]),
    ("Physical Security", [
        "Office access requires an individually issued badge. Badges must not be shared and "
        "tailgating through controlled doors is prohibited.",
        "Visitors must be registered in advance, sign in at reception and be escorted at all "
        "times while inside secure areas.",
        "A clean desk is required: documents classified Confidential or Restricted must be "
        "locked away when the desk is unattended.",
    ]),
]

PROCUREMENT: list[Section] = [
    ("Purpose and Scope", [
        "This Finance and Procurement Policy, reference FIN-2025-01, governs how Northwind "
        "Analytics commits to and pays for goods and services. It applies to all employees "
        "who request, approve or receive purchases.",
        "The policy is owned by the Chief Financial Officer and is reviewed every 12 months. "
        "It does not cover employee expense claims, which are described in the Employee "
        "Handbook.",
    ]),
    ("Purchase Approval Thresholds", [
        "A purchase order is required for any commitment above 1000 US dollars. Purchases at "
        "or below 1000 US dollars may be made on a corporate card without a purchase order.",
        "Commitments up to 10000 US dollars are approved by the budget owner. Commitments "
        "from 10000 up to 50000 US dollars require approval from the department Vice "
        "President.",
        "Any commitment above 50000 US dollars requires approval from the Chief Financial "
        "Officer. Commitments above 250000 US dollars additionally require board "
        "notification.",
        "Splitting a single commitment into smaller purchases in order to stay below an "
        "approval threshold is a disciplinary matter.",
    ]),
    ("Supplier Onboarding", [
        "New suppliers must complete the supplier onboarding form and provide bank details "
        "through the verified supplier portal. Bank details are never accepted by email.",
        "A change to existing supplier bank details requires verbal confirmation with a "
        "known contact at the supplier using a previously recorded telephone number.",
        "Suppliers processing Restricted data must additionally clear the security review "
        "described in the Information Security Policy before a contract is signed.",
    ]),
    ("Invoices and Payment Terms", [
        "Standard payment terms are 30 days from the date of a valid invoice. Terms shorter "
        "than 30 days require approval from the Chief Financial Officer.",
        "Invoices must quote a valid purchase order number where one was required. Invoices "
        "without a purchase order number are returned to the supplier unpaid.",
        "The finance team runs two payment cycles per month, on the 10th and on the 25th.",
        "Disputed invoices must be raised with the finance team within 14 days of receipt.",
    ]),
    ("Corporate Cards", [
        "Corporate cards are issued to employees who make recurring purchases on behalf of "
        "the company. The default monthly limit is 5000 US dollars.",
        "Card transactions must be reconciled with a receipt within 15 days of the "
        "transaction date. Cards with unreconciled transactions older than 45 days are "
        "suspended.",
        "Personal use of a corporate card is prohibited, including where the employee "
        "intends to repay the amount.",
    ]),
    ("Travel Booking", [
        "All business travel must be booked through the approved travel management company. "
        "Direct bookings are reimbursed only where the travel management company was unable "
        "to provide an equivalent option.",
        "Travel must be booked at least 14 days in advance where the itinerary is known. "
        "Bookings made inside 7 days require manager approval.",
        "Rental cars are limited to intermediate class or below. Ride hailing is preferred "
        "for journeys under 50 kilometres.",
    ]),
    ("Budget Management", [
        "Budget owners review actual spend against budget monthly. A forecast variance above "
        "10 percent must be explained in the monthly finance review.",
        "Unspent departmental budget does not carry over between financial years. The "
        "financial year runs from 1 February to 31 January.",
    ]),
]

CONTRACTOR_MD = """# Contractor Engagement Policy

This policy covers independent contractors and agency workers engaged by Northwind
Analytics. Contractors are not employees and the Employee Handbook does not apply to them.

## Engagement and Statement of Work

Every contractor engagement requires a signed statement of work that specifies deliverables,
the daily rate and an end date. Open-ended contractor engagements are not permitted.

The maximum initial engagement length is 6 months. An engagement may be extended twice, to a
total maximum of 18 months, after which a break of at least 3 months is required.

## Rates and Invoicing

Contractor daily rates are agreed in the statement of work and are inclusive of all taxes.
Rate changes take effect only at the start of a new statement of work.

Contractors invoice monthly in arrears. Invoices are paid within 21 days of receipt,
which is shorter than the standard supplier payment terms.

Contractors must submit a timesheet approved by their engagement manager before invoicing.
Timesheets submitted more than 30 days after the end of the month may be rejected.

## Leave and Working Arrangements

Contractors do not accrue paid annual leave, paid sick leave or parental leave. Any period
of non-availability must be agreed with the engagement manager at least 5 working days in
advance.

Contractors determine their own working hours unless the statement of work specifies
coverage requirements. Contractors are not required to observe core collaboration hours.

## Expenses

Contractor expenses are reimbursed only where the statement of work explicitly allows them.
Where allowed, expenses must be submitted with the monthly invoice and are capped at 500 US
dollars per month unless pre-approved in writing.

Contractors are not eligible for the meal per diem, the home office stipend or the learning
budget available to employees.

## Access and Security

Contractors receive a named account and are subject to the Information Security Policy in
full. Contractor accounts are time limited and expire automatically on the statement of work
end date.

Contractors may not use personal devices to access Confidential or Restricted data. A
managed company device is issued where such access is required.

## Termination

Either party may terminate a contractor engagement with 10 working days written notice. The
notice period for employees is longer and is defined in the Employee Handbook.

Northwind Analytics may terminate an engagement immediately for a material breach of the
Information Security Policy or the Business Conduct section of the handbook.

## Intellectual Property

All intellectual property created by a contractor in the course of an engagement is assigned
to Northwind Analytics on creation. This assignment survives the end of the engagement.
"""

RETENTION_MD = """# Data Retention Schedule

This schedule defines how long Northwind Analytics keeps each category of data. It
implements the retention requirements of the Information Security Policy and is reviewed
annually by the Data Protection Officer.

## General Principles

Data is kept only as long as there is a lawful basis and a business need. When a retention
period ends, data is deleted or irreversibly anonymised within 30 days.

A legal hold overrides every retention period in this schedule. Data under legal hold is
retained until the hold is formally lifted by the legal team.

## Customer Data

Active customer account data is retained for the duration of the contract.

After a contract ends, customer account data is retained for 6 years to satisfy statutory
accounting and limitation periods, then deleted.

Customer support tickets are retained for 3 years from the date the ticket is closed.

Product usage telemetry linked to an identifiable user is retained for 13 months. Aggregated
telemetry that cannot be linked to an individual is retained indefinitely.

## Employee and Recruitment Data

Employee personnel files are retained for 7 years after the end of employment.

Unsuccessful candidate applications are retained for 12 months after the hiring decision,
unless the candidate consents to a longer talent pool retention of 24 months.

Right to work documentation is retained for 2 years after employment ends.

## Security and System Data

Security relevant logs are retained for 400 days, consistent with the Information Security
Policy.

Application debug logs are retained for 30 days. Debug logs must not contain personal data
or authentication secrets.

Video surveillance recordings from office premises are retained for 31 days and are
accessible only to the physical security team.

Backups are retained for 35 days. A deletion request is applied to production systems
immediately and propagates through backups within 35 days.

## Financial Records

Invoices, purchase orders and payment records are retained for 10 years from the end of the
financial year to which they relate.

Corporate card statements and receipts are retained for 7 years.

## Deletion Requests

A verified data subject deletion request is completed within 30 days. Where the request
cannot be completed within 30 days, the data subject is informed of the delay and the reason
before the deadline.
"""

RELEASE_MD = """# Product Release Process

This document describes how Northwind Analytics ships product changes. It complements the
Engineering Onboarding Guide, which covers day-to-day development workflow.

## Release Trains

Production releases run on a fixed schedule twice per week, on Tuesday and Thursday at 13:00
UTC. A release train departs whether or not a given change is ready; changes that miss a
train wait for the next one.

Hotfixes may be released outside the train schedule. A hotfix requires approval from the
service owner and an incident or ticket reference.

## Versioning

Services follow semantic versioning. A breaking change to a public API requires a major
version increment and a deprecation notice published at least 90 days in advance.

Client libraries support the two most recent major versions. Older major versions receive
security fixes only.

## Feature Flags

Every user-visible change ships behind a feature flag. Flags default to off in production
and are enabled progressively.

The standard rollout is 5 percent of traffic, then 25 percent, then 100 percent, with at
least 24 hours between stages.

Feature flags must be removed within 60 days of reaching full rollout. The platform team
reports stale flags weekly.

## Release Readiness

A change is release ready when the pipeline is green, the feature flag configuration is
documented, and a rollback path is described in the change ticket.

Changes that alter a database schema require a migration reviewed by a database owner.
Backward incompatible migrations must be split into at least two releases.

## Rollback

Any release can be rolled back by the on-call engineer without further approval. The target
rollback time is under 10 minutes.

A rollback is always preferred over a forward fix during an active incident.

## Release Communication

The release notes are published to the customer changelog on the day of the release.

Changes that affect the public API or pricing require at least 30 days notice to customers
before the release.
"""
