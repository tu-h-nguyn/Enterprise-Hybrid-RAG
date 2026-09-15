# Product Release Process

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
