# FPL Squad Advisor

A hybrid data-science + multi-agent system for building and managing an FPL squad across the 2026/27 Premier League season.

See `PROJECT_PLAN.md` for the full architecture and reasoning. Short version: a deterministic pipeline (data → expected-points model → ILP optimizer) picks the squad, while a layer of specialist agents (per position, plus fixtures and chip-strategy agents) feed it qualitative research an optimizer alone can't produce. A manager agent consolidates everything — including Neil's own fan judgment, captured in `docs/player_judgments.md` and `docs/football_domain_knowledge.md` — into a final recommendation.

## Status
Scaffolding in progress. See `PROJECT_PLAN.md` for the build sequence.

## Structure
```
data/                  raw + processed FPL data
models/                expected-points model
optimization/          squad/XI/transfer/chip optimizer
agents/                position, fixtures, chip-strategy, and orchestrator agents
reports/               weekly recommendation output
docs/
  football_domain_knowledge.md   general game-reading heuristics (living doc)
  player_judgments.md            specific fan calls on players (living doc)
PROJECT_PLAN.md         full architecture and build plan
```

## Setup
```
pip install -r requirements.txt
```
