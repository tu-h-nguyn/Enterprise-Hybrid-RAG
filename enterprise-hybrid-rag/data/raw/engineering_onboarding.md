# Engineering Onboarding Guide

Welcome to the Northwind Analytics engineering organisation. This guide covers your first
two weeks. It is maintained by the Developer Experience team.

## Week One Checklist

On day one you receive your laptop, single sign-on credentials and a hardware security key.
Your buddy is assigned automatically and will meet you daily for the first week.

By the end of week one you should have shipped one small pull request to production. This is
deliberate: the onboarding target is a first merged pull request within 5 working days.

## Development Environment

All services target Python 3.11 and Node 20. Docker Desktop or an equivalent container
runtime is required.

Run `make bootstrap` in the monorepo root. It installs dependencies, seeds a local Postgres
database and starts the supporting containers. A full bootstrap takes about 12 minutes on a
current laptop.

Local secrets are read from `.env.local`, which is never committed. Production secrets live
in the secrets manager and are never available on developer machines.

## Git Workflow

We use trunk-based development. The `main` branch is protected and always deployable.

Branch names follow the pattern `feat/ENG-1234-short-description`, using `feat/`, `fix/` or
`chore/` followed by the ticket identifier. Merges into `main` are squash merges.

## Code Review

Pull requests should change no more than 400 lines. Larger changes must be split unless the
change is a mechanical refactor.

Service code requires 2 approvals before merge. Documentation-only changes require 1
approval. The author must not approve their own pull request.

Reviewers are expected to give a first response within 1 business day.

## Continuous Integration and Deployment

The pipeline runs five stages in order: lint, unit tests, integration tests, build, and
deploy to staging. Deployment to production requires a manual approval from a service owner.

New code must maintain at least 80 percent line coverage. The pipeline fails below that
threshold.

Staging deployments happen automatically on every merge to `main`. Production deployments
are batched and released twice per week, on Tuesday and Thursday.

## On-Call

Each service team runs a weekly on-call rotation managed in PagerDuty. The handover happens
every Wednesday at 14:00 UTC.

The acknowledgement target for a paging alert is 5 minutes. Every alert must link to a
runbook stored under the `/runbooks` directory of the service repository.

On-call engineers receive a compensation allowance of 250 US dollars per completed week of
rotation.

## Platform Architecture

The platform consists of four primary services: `ingestion-api`, `feature-store`,
`model-gateway` and `dashboard-web`.

Persistent state lives in Postgres 15. Redis is used for caching and rate limiting. Kafka
carries event streams between the ingestion API and the feature store.

The model gateway is the only component permitted to call external model providers. Direct
provider calls from other services are blocked at the network layer.

## Glossary

Feature drift refers to a change in the statistical distribution of a model input relative
to its training distribution. Drift is monitored daily and alerts at a population stability
index above 0.2.
