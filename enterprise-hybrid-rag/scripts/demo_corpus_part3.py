"""Third corpus module: four documents chosen to create realistic confusion.

Each one shares heavy vocabulary with an existing document while stating
*different* facts:

* ``it_service_desk_guide``    reuses the P1-P4 priority language of the
  customer SLA, with different response targets. A retriever that matches on
  "P1 response time" alone will get the wrong document.
* ``hr_recruitment_process``   overlaps the handbook (referral bonus) and the
  retention schedule (candidate data).
* ``model_governance_policy``  overlaps the onboarding glossary (feature drift)
  and the security policy (approval, review cadence).
* ``office_facilities_guide``  overlaps the security policy's physical section.
"""

from __future__ import annotations

IT_SERVICE_DESK_MD = """# IT Service Desk Guide

This guide explains how employees request help from the internal IT service desk. It covers
internal support only. Commitments made to paying customers are defined in the Customer
Support Service Level Agreement and are not affected by this guide.

## Contacting the Service Desk

The service desk is reached through the internal help portal or the `#it-help` channel.
Telephone support is available only for a total loss of access.

The service desk operates 07:00 to 19:00 Central European Time on working days. Outside
these hours only P1 tickets are handled, by the on-call IT engineer.

## Internal Ticket Priorities

An internal P1 is a total loss of service affecting an entire office or a business critical
system. The internal first response target for a P1 is 15 minutes.

An internal P2 affects a team or blocks a critical deadline. The response target is 2
working hours.

An internal P3 is a single user problem with a workaround available. The response target is
1 working day.

An internal P4 covers requests, questions and improvements. The response target is 5 working
days.

These internal targets are separate from the customer facing targets and are measured only
during service desk operating hours.

## Hardware Requests

Standard hardware, including laptops, monitors and docking stations, is delivered within 5
working days of an approved request.

Non-standard hardware requires a justification and approval from the department budget
owner. Delivery for non-standard hardware is typically 15 working days.

A replacement for failed hardware is issued within 1 working day where a loan pool device is
available.

## Software and Licence Requests

Software on the approved catalogue is installed automatically after the request is
submitted. No manual approval is needed.

Software that is not on the catalogue requires both a security review and a licence cost
approval from the budget owner. The security review takes up to 10 working days.

Employees must not install unapproved software, including browser extensions, on managed
devices.

## Access and Account Problems

A forgotten password is resolved through self service password reset. The service desk
cannot reset a password on request without identity verification by video call.

A locked account unlocks automatically after 30 minutes. The service desk can unlock it
sooner after identity verification.

Requests for access to a new system are raised as an access request, not as an incident, and
follow the approval path defined in the Information Security Policy.

## Equipment Return

Departing employees return equipment to the service desk on or before their final working
day. Equipment not returned within 10 working days after the final day is deducted from the
final salary payment where local law permits.
"""

HR_RECRUITMENT_MD = """# Recruitment and Hiring Process

This document describes how Northwind Analytics hires permanent employees. Contractor
engagements follow the Contractor Engagement Policy instead.

## Opening a Role

A role is opened only against an approved headcount plan. The hiring manager submits a role
brief containing the scope, the level and the target salary band.

The role brief is approved by the department Vice President and by People Operations. The
approval target is 5 working days.

## Interview Loop

The standard interview loop has four stages: a recruiter screen, a hiring manager interview,
a technical or craft assessment, and a values interview.

The recruiter screen lasts 30 minutes. The hiring manager interview and the values interview
each last 45 minutes. The craft assessment lasts 90 minutes.

Every interviewer submits written feedback within 24 hours of the interview and before
seeing the feedback of other interviewers. Feedback submitted late is still recorded but is
flagged in the hiring review.

A candidate must not be interviewed by their prospective direct reports.

## Take-Home Assessments

Where a take-home assessment is used instead of a live craft assessment, it must be designed
to take no more than 3 hours. Candidates are given at least 5 calendar days to complete it.

Candidates who complete a take-home assessment receive written feedback whether or not they
progress.

## Hiring Decision and Offer

The hiring decision is made in a debrief attended by all interviewers. A hire decision
requires consensus; a single strong objection blocks the hire.

Offers within the approved salary band are approved by the hiring manager and People
Operations. Offers above the band require Chief People Officer approval.

The candidate receives the written offer within 2 working days of the debrief. Offers remain
open for 10 calendar days unless extended by agreement.

## Referrals

Employee referrals enter the process at the recruiter screen stage and receive a response
within 5 working days.

The referral bonus is paid after the referred employee completes six months of service, as
described in the Employee Handbook. A referral bonus is not paid where the referrer is the
hiring manager for the role.

## Candidate Data

Candidate records are stored in the applicant tracking system. Interview feedback is visible
to the hiring team only.

Retention of unsuccessful candidate data follows the Data Retention Schedule. Candidate data
must never be copied into spreadsheets or shared outside the applicant tracking system.
"""

MODEL_GOVERNANCE_MD = """# Model Governance Policy

This policy governs how machine learning models are approved, deployed and monitored at
Northwind Analytics. It applies to every model whose output is visible to a customer or used
to make an automated decision.

## Model Risk Tiers

Models are assigned to one of three risk tiers. Tier 1 models make or materially influence
decisions about individuals. Tier 2 models affect customer-visible product behaviour. Tier 3
models are internal only.

A Tier 1 model requires approval from the Model Review Board before production deployment. A
Tier 2 model requires approval from the service owner and a documented evaluation. A Tier 3
model requires only a documented evaluation.

## Model Review Board

The Model Review Board meets fortnightly. It is chaired by the Head of Data Science and
includes a representative from legal, security and product.

A review submission must be filed at least 5 working days before the meeting and must
include the model card, the evaluation report and the intended rollout plan.

## Model Cards

Every production model has a model card recording its purpose, training data sources,
evaluation metrics, known limitations and the owning team.

The model card is updated on every retraining and is reviewed at least every 6 months.

## Evaluation Requirements

Every model is evaluated on a held-out test set that was not used during training or model
selection.

Tier 1 models additionally require a subgroup performance analysis across the protected
attributes relevant to the use case. A performance gap above 5 percentage points between
subgroups must be documented and explicitly accepted by the Model Review Board.

Evaluation results are stored with the model version. A model cannot be promoted to
production without a stored evaluation.

## Monitoring and Drift

Production models are monitored daily for input drift and for prediction distribution
shifts.

An alert is raised when the population stability index exceeds 0.2, which matches the
threshold described in the Engineering Onboarding Guide.

A Tier 1 model that triggers a drift alert must be reviewed by its owning team within 2
working days.

## Retraining and Rollback

Scheduled retraining runs quarterly for Tier 1 and Tier 2 models unless the model card
specifies a different cadence.

A retrained model is deployed only after it matches or exceeds the current production model
on the primary evaluation metric.

Any model deployment can be rolled back by the on-call engineer. The previous model version
is retained and deployable for at least 90 days.

## Third-Party and Foundation Models

Use of an external model provider requires a vendor security assessment and a data
processing agreement before any company data is sent.

Confidential or Restricted data may not be sent to an external model provider unless the
contract explicitly prohibits training on that data.
"""

OFFICE_FACILITIES_MD = """# Office and Facilities Guide

This guide covers the practical use of Northwind Analytics offices. Security requirements
for physical access are defined in the Information Security Policy and take precedence over
anything in this guide.

## Opening Hours and Access

Offices are open to badge holders from 06:00 to 22:00 on working days. Access outside these
hours requires approval from the office manager and is logged.

Weekend access is available to badge holders but the reception desk is unstaffed and
deliveries cannot be accepted.

## Desk Booking

Desks are booked through the workplace application. Bookings open 14 days in advance and are
released automatically if the desk is not claimed by 10:30 on the day.

Teams may reserve a block of desks for a fixed team day once per week. Permanent desk
assignments are available only where an accessibility requirement or specialist equipment
makes them necessary.

## Meeting Rooms

Meeting rooms are booked through the calendar system. Rooms with more than 8 seats may not
be booked for meetings of fewer than 4 people during core hours.

A room booking is released automatically if nobody checks in within 10 minutes of the start
time.

## Visitors

Visitors are registered in advance through the workplace application and sign in at
reception on arrival. Every visitor is escorted while inside the office.

Visitors receive a temporary badge that is valid for the day only and must be returned at
reception on departure.

## Deliveries and Post

Personal deliveries to the office are discouraged and are accepted only during reception
hours of 09:00 to 17:00.

Incoming post is sorted daily. Post that remains uncollected for 30 days is returned to the
sender or securely destroyed.

## Kitchen and Catering

Kitchens are stocked daily. Catering for meetings of 10 people or more is ordered through
the office manager with at least 3 working days notice.

Employees are responsible for clearing their own items from shared fridges, which are
emptied every Friday evening.

## Health and Safety

Fire evacuation drills are held twice per year in each office. Every employee must complete
a workstation assessment within 30 days of joining.

First aid kits and defibrillators are located at each reception. Accidents and near misses
are reported through the workplace application on the day they occur.

## Parking and Travel

Parking spaces are limited and allocated by lottery each quarter. Accessible parking spaces
are reserved and are outside the lottery.

Secure bicycle storage and showers are available at every office. A cycle to work allowance
of 300 US dollars per year is available to all employees.
"""
