# Provenance

## Snapshot type

This repository was created on 2026-08-28 as a new, no-history, clean-room public
portfolio project.

No source file, test, documentation file, Git object, issue, pull request,
workflow, dataset, research artifact, strategy configuration, deployment asset,
or visual asset was copied from the private CCB repository.

## Intellectual sources

The statistical concepts were implemented from the public research cited in
docs/methodology.md:

- Bailey and López de Prado, The Deflated Sharpe Ratio.
- Bailey, Borwein, López de Prado, and Zhu, The Probability of Backtest
  Overfitting.

The implementation uses Python and NumPy APIs. It does not copy RiskLabAI, pypbo,
qnt, exchange SDK, frontend component, logo, or vendor-data source code.

## Relationship to the private project

The public project demonstrates general engineering lessons from a larger private
quantitative research system: explicit cost accounting, synchronized trial
matrices, statistical evidence checks, deterministic fixtures, and strict
research-to-execution separation.

Private thresholds, schemas, Registry bindings, candidate identities, trial
windows, negative-result inventories, operational configuration, and execution
paths are intentionally absent.

## Review rule

Future contributions must document any copied or derived source before merge.
Third-party code must retain its required copyright and license notice. Data must
have a documented redistribution right; otherwise use synthetic fixtures.
