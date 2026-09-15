# Data Engineering and Pipeline Standards

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
