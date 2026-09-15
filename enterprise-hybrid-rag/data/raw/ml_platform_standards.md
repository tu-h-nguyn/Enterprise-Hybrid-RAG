# Machine Learning Platform Standards

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
