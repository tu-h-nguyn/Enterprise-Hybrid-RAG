# Production Incident Response Runbook

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
