# Business Continuity and Disaster Recovery Plan

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
