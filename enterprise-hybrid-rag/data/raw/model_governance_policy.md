# Model Governance Policy

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
