# FPL Squad Advisor — Project Plan

## Purpose
A portfolio project connecting data science and AI orchestration to sports analytics: an ongoing tool that builds and manages an FPL squad for the 2026/27 Premier League season (starts Fri 21 Aug 2026).

Dual goal:
1. Demonstrate applied data science (predictive modeling + constrained optimization).
2. Demonstrate agentic AI orchestration (specialist subagents + a manager agent) — the skill most relevant to breaking into AI strategy/business ops roles.

Football is not baseball — outcomes are noisy, low-scoring, and driven by qualitative factors (rotation, tactics, morale) that don't reduce cleanly to stats. The system is deliberately **not** fully data-driven: it has an explicit channel for Neil's own football judgment to override or adjust the model.

## Season context (as of Aug 2026)
- 2026/27 season starts Fri 21 Aug 2026 (Arsenal–Coventry opener), ends 30 May 2027.
- Promoted: Coventry City, Ipswich Town, Hull City (no current top-flight data — need fallback handling).
- Relegated: Wolverhampton Wanderers, Burnley, West Ham United.
- DefCon (defensive contribution) scoring rule unchanged from last season. BPS formula recalibrated slightly to favor holding midfielders over full-backs.

## Architecture — 4 layers

### 1. Data layer (deterministic)
- Official FPL API: bootstrap-static (players/prices/positions), fixtures, element-summary (per-player history).
- Historical multi-season data (vaastav's public FPL archive) for model training/backtesting.
- Underlying stats (xG/xA) where available.
- Custom fixture-difficulty rating (FPL's own FDR is simplistic).
- Fallback handling for promoted clubs with no PL-level data.

### 2. Model layer (deterministic)
- Expected points (xP) per player per gameweek: per-90 stats, start probability, fixture difficulty, FPL scoring rules (goals/assists/clean sheets by position, DefCon, saves, bonus).
- Start with a transparent statistical formula (explainable); ML model (e.g. gradient boosting) as a later stretch goal, validated by backtest.

### 3. Optimization layer (deterministic)
- ILP (PuLP) squad selection: budget £100m, 2 GK/5 DEF/5 MID/3 FWD, max 3 per club.
- Starting XI + captain solve (valid formation).
- Weekly transfer optimizer (accounts for -4 hit cost).
- Rule-based chip-timing flags (feeds from the chip strategy agent below).

### 4. Agent layer (the showcase piece)
Specialist agents research qualitative signals the model can't capture; the manager agent consolidates everything and the optimizer still makes the final numeric call. This is deliberate: agents inform inputs, they don't freehand the squad.

- **GK agent** — clean sheet probability, save volume, penalty-save duty, distribution role.
- **DEF agent** — clean sheets + attacking returns, set-piece threat, DefCon nailing-on.
- **MID agent** — xG/xA, penalty/set-piece duty, minutes security, DefCon (holding mids).
- **FWD agent** — xG, penalty duty, rotation risk, minutes security.
- **Fixtures agent** — upcoming fixture swings, congestion (Europe/cup distractions), double/blank gameweeks, team form trajectory. Feeds the position agents and the chip agent.
- **Chip strategy agent** — Wildcard/Bench Boost/Triple Captain/Free Hit timing, using fixtures agent output.
- **Orchestrator (manager) agent** — consolidates all subagent outputs + Neil's `docs/player_judgments.md` overrides, adjusts xP accordingly, runs the optimizer, and writes a plain-English decision memo (squad, XI, captain/vice, transfers, chip advice, reasoning).

### Fan judgment channel (not fully data-driven, by design)
Two living docs capture Neil's own football knowledge and specific calls, and both are explicit inputs to the orchestrator:
- `docs/football_domain_knowledge.md` — general heuristics for how to read the game (rotation, set pieces, fixture congestion, promoted-team dynamics, etc.), captured as we build.
- `docs/player_judgments.md` — specific fan opinions on individual players that adjust or override model output.

## Model to use for build phases
Scaffolding and data engineering: current model is fine. Once we're into the xP model, optimizer, and agent logic (the genuinely hard reasoning/code-correctness work), switch to **Opus** for subagent dispatches on those phases — it's the stronger fit for careful, high-stakes technical work versus Fable, which leans more creative/persona-driven. If Neil wants the *main* conversation on Opus too, that's a model-picker setting in the app itself, not something controllable from inside the session.

## Build sequence
1. Scaffolding & docs system
2. FPL data pipeline
3. xP model + backtest
4. Optimizer
5. Position agents (GK/DEF/MID/FWD)
6. Fixtures agent
7. Chip strategy agent
8. Orchestrator agent
9. End-to-end GW1 recommendation run
10. Weekly automation (scheduled task)
11. Verification & backtesting

## Deliverables
- Documented Python codebase (data/models/optimization/agents/reports), GitHub-ready.
- Weekly recommendation report (transfers, captain, chip advice, reasoning).
- README + this plan doc as the portfolio narrative.
