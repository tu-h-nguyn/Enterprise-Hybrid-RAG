"""Additional demo documents (part 4).

These extend the Northwind Analytics corpus so the evaluation has enough
material to discriminate between retrieval strategies. They deliberately reuse
vocabulary that already appears elsewhere ("approval", "30 days", "escalation",
"retention", "notice period") so that lexical and semantic retrievers face
genuine distractors rather than a corpus where every question has exactly one
plausible document.
"""

from __future__ import annotations

Section = tuple[str, list[str]]

TRAVEL_MD = """# Travel and Expense Policy

This policy applies to all Northwind Analytics employees and contractors who incur costs on
company business. It is owned by the Finance team and takes effect from 1 March 2025 under
document reference FIN-TE-4.

## Booking and Approval

All travel must be booked through the corporate travel portal at least 14 days before
departure. Bookings made inside 14 days require written approval from a department director.

Economy class is the standard for all flights under six hours of scheduled flying time.
Premium economy is permitted for flights of six hours or more. Business class requires
approval from a member of the executive team and is never permitted for domestic travel.

Rail travel is preferred over air travel for any journey where the total rail time is under
five hours. Standard class is the default; first class rail is not reimbursed.

## Accommodation

The nightly accommodation cap is 180 euro in Berlin, Munich and Frankfurt, 220 euro in
London and Zurich, and 140 euro in all other locations. Caps exclude local city taxes.

Employees who stay with friends or family instead of booking a hotel may claim a flat
allowance of 25 euro per night without receipts.

## Per Diem and Meals

The daily meal allowance is 45 euro for full travel days and 22 euro for partial travel days
of less than eight hours. The allowance is reduced by 15 euro for each meal provided by a
conference or client.

Alcohol is not reimbursable except at client entertainment events that have been approved in
advance by a director.

## Ground Transport

Mileage for private cars is reimbursed at 0.38 euro per kilometre. The first 20 kilometres of
any journey that starts at the employee's home are treated as commuting and are not
reimbursed.

Taxi and ride-hailing journeys are reimbursed where public transport is unavailable, unsafe
or would add more than 45 minutes to the journey.

## Receipts and Submission

Itemised receipts are required for every expense above 25 euro. Expenses below that
threshold may be claimed with a written description.

Expense claims must be submitted within 45 days of the expense being incurred. Claims
submitted after 45 days require a written exception from the Head of Finance and are paid in
the following quarter.

Reimbursement is paid with the next scheduled payroll run, provided the claim is approved at
least five working days before the payroll cut-off.

## Non-Reimbursable Items

The following are never reimbursed: traffic fines and parking penalties, personal travel
insurance, airline lounge memberships, in-flight entertainment purchases, hotel minibar
charges, and upgrades paid directly at check-in.
"""

REMOTE_MD = """# Remote and Hybrid Work Policy

Northwind Analytics operates a hybrid model. This policy sets out eligibility, equipment and
tax obligations. It is reference HR-RW-2 and replaces the interim guidance issued in 2023.

## Eligibility

All permanent employees who have completed their probation period are eligible for hybrid
work. Probation is six months for all roles except engineering, where it is three months.

Employees in customer-facing operational roles must be present in an office for a minimum of
two days per week. All other employees must attend at least four days per calendar month.

Fully remote arrangements require approval from both the line manager and the Head of People
Operations and are reviewed every twelve months.

## Home Office Equipment

Each remote employee receives a one-off home office budget of 900 euro, refreshed every
three years. The budget covers a desk, chair, monitor and peripherals.

Laptops are provided by the company and are replaced on a four-year cycle, or earlier if a
hardware fault cannot be repaired within five working days.

Equipment purchased with the home office budget remains company property and must be
returned within 30 days of the end of employment.

## Connectivity Allowance

Remote employees receive a monthly connectivity allowance of 40 euro. The allowance is paid
through payroll and is treated as taxable income in most jurisdictions.

The company does not reimburse the cost of upgrading a residential internet connection
beyond the standard allowance.

## Working from Abroad

Employees may work from another country for up to 20 working days per calendar year without
a formal assignment, provided the country is on the approved list maintained by the Legal
team.

Requests to work abroad must be submitted at least 30 days in advance because permanent
establishment and payroll tax exposure must be assessed for each country.

Working from a country that is subject to sanctions or export controls is prohibited without
exception.

## Health and Safety

Remote employees must complete the home workstation assessment within 14 days of starting a
hybrid arrangement and must repeat it every two years.

The company's employer liability insurance covers the designated home workspace only during
agreed working hours.
"""

INCIDENT_MD = """# Production Incident Response Runbook

This runbook governs how Northwind Analytics responds to production incidents affecting
customer-facing systems. It is owned by the Site Reliability Engineering team, reference
SRE-IR-9.

## Severity Levels

A SEV1 incident is a total loss of service, a confirmed data loss, or an active security
breach affecting customer data.

A SEV2 incident is a partial loss of service or severe degradation affecting more than 20
percent of customers.

A SEV3 incident is a degradation with a workaround, or a failure affecting a single customer
or a non-critical subsystem.

## On-Call Rotation

The primary on-call engineer must acknowledge a page within 5 minutes. If the page is not
acknowledged, it escalates automatically to the secondary on-call engineer after 10 minutes.

On-call shifts run for seven days, starting Wednesday at 10:00. No engineer may be scheduled
for more than one primary shift in any 28-day period.

On-call engineers receive a standby allowance of 250 euro per week, plus compensatory time
off for any callout occurring between 22:00 and 07:00.

## Declaring an Incident

Any engineer may declare an incident. There is no penalty for declaring an incident that
later turns out to be minor; under-declaration is treated as the more serious failure.

Declaring an incident automatically creates a dedicated channel, a timeline document and an
incident commander assignment.

## Roles

The incident commander owns decision-making and is explicitly not expected to debug. The
communications lead owns customer and internal updates. The operations lead owns the
technical investigation.

For a SEV1, the incident commander must be someone other than the engineer performing the
remediation.

## Communication

Internal status updates are posted every 30 minutes during a SEV1 and every 60 minutes
during a SEV2.

The public status page must be updated within 15 minutes of a SEV1 being declared. Customer
notification for a confirmed data breach follows the timelines in the data processing
agreement, not this runbook.

## Postmortems

Every SEV1 and SEV2 requires a written postmortem. The draft is due within 5 working days of
resolution and the review meeting must take place within 10 working days.

Postmortems are blameless: they describe systems and decisions, not individuals. Action items
must have a named owner and a due date, and are tracked to completion in the engineering
backlog.

Any action item arising from a SEV1 that remains open after 60 days is escalated to the VP of
Engineering.
"""

BCP_MD = """# Business Continuity and Disaster Recovery Plan

This plan describes how Northwind Analytics restores service after a major disruption. It is
reference BCP-3.1 and is tested twice a year.

## Recovery Objectives

The recovery time objective for the core analytics platform is 4 hours. The recovery point
objective is 15 minutes.

For internal corporate systems such as the intranet and the expense portal, the recovery time
objective is 3 working days and the recovery point objective is 24 hours.

## Backup Schedule

Transactional databases are backed up continuously through write-ahead log shipping, with a
full snapshot taken every 6 hours.

Object storage is replicated to a secondary region within 15 minutes of write. Backups are
retained for 35 days for operational recovery and 7 years for financial records.

Backup restoration is tested quarterly on a randomly selected dataset. A restoration test
that fails must be repeated successfully within 10 working days.

## Failover

The platform runs active-passive across two regions in the European Union. Failover is
initiated by the incident commander and requires confirmation from one other member of the
SRE team.

Expected failover time is 25 minutes, of which approximately 10 minutes is DNS propagation.

## Crisis Management

The crisis management team is convened for any disruption expected to exceed 4 hours. It is
chaired by the Chief Operating Officer.

If the primary office is inaccessible, staff work remotely by default. The alternate meeting
location is the co-working facility contracted in each city.

## Testing and Review

A full failover exercise is conducted every six months. A tabletop exercise is conducted in
the intervening quarter.

The plan is reviewed annually, and after every SEV1 incident that triggered a regional
failover.
"""

DPA_MD = """# Customer Data Processing Agreement Summary

This is the internal summary of the standard data processing agreement that Northwind
Analytics signs with customers. The signed agreement prevails in any conflict. Reference
LEG-DPA-5.

## Roles of the Parties

The customer is the data controller. Northwind Analytics acts as data processor and
processes personal data only on documented instructions from the customer.

Where Northwind Analytics determines the purposes of processing, for example for product
telemetry and billing, it acts as an independent controller.

## Subprocessors

A current list of subprocessors is published on the trust portal. Customers are notified at
least 30 days before a new subprocessor is added.

A customer may object to a new subprocessor within 14 days of notification on reasonable data
protection grounds. If the objection cannot be resolved, the customer may terminate the
affected service without penalty.

## Security Measures

Personal data is encrypted in transit using TLS 1.2 or higher and at rest using AES-256.

Access to customer production data requires a documented business reason, is granted for a
maximum of 8 hours at a time, and is logged immutably.

## Breach Notification

Northwind Analytics notifies the customer without undue delay and in any event within 48
hours of becoming aware of a personal data breach affecting that customer's data.

The notification includes the nature of the breach, the categories and approximate number of
records affected, the likely consequences and the measures taken.

Notification to supervisory authorities is the responsibility of the customer as controller,
except where Northwind Analytics acts as controller.

## International Transfers

Personal data is stored in the European Union by default. Transfers outside the European
Economic Area rely on the European Commission standard contractual clauses together with a
transfer impact assessment.

Customers on the Enterprise plan may request EU-only processing, including support access,
for an additional fee.

## Deletion and Return

On termination, customer data is returned in a machine-readable format on request and is
deleted from production systems within 30 days and from backups within 90 days.

A certificate of deletion is provided on written request within 15 working days of
completion.
"""

VENDOR_MD = """# Vendor Risk Assessment Procedure

This procedure describes how Northwind Analytics assesses and monitors third-party vendors.
It is owned jointly by Procurement and Security, reference SEC-VR-6.

## Vendor Tiers

A Tier 1 vendor processes customer personal data or has privileged access to production
systems.

A Tier 2 vendor processes internal company data or has access to corporate systems but not
to production.

A Tier 3 vendor provides goods or services with no access to company data, for example office
catering or facilities maintenance.

## Assessment Requirements

Tier 1 vendors require a full security assessment, a review of an independent audit report
such as SOC 2 Type II or ISO 27001, a data protection impact assessment, and a signed data
processing agreement before any contract is executed.

Tier 2 vendors require a completed security questionnaire and evidence of an information
security policy.

Tier 3 vendors require only standard commercial due diligence.

## Reassessment Cadence

Tier 1 vendors are reassessed every 12 months. Tier 2 vendors are reassessed every 24 months.
Tier 3 vendors are reviewed at contract renewal.

An unscheduled reassessment is triggered by a publicly disclosed breach at the vendor, a
change of ownership, or a material change in the services provided.

## Exceptions

A time-limited exception may be granted by the Chief Information Security Officer for a
maximum of 90 days where a business need is urgent and compensating controls are documented.

Exceptions are recorded in the risk register and reported to the Audit Committee each
quarter.

## Offboarding

When a vendor contract ends, access is revoked within 24 hours and confirmation of data
deletion is obtained within 60 days.
"""
