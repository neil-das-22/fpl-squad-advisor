# Research Agent Specs

This is the living spec for the six research agents dispatched each gameweek
(currently run live via Claude subagents in Cowork, per PROJECT_PLAN.md).
Until now these only existed as one-off prompts written inline each time
they were dispatched -- this file makes them a persistent, editable artifact
so they can be reviewed and tweaked deliberately instead of re-invented each
run. Whoever (or whatever) dispatches these agents each week should pull the
prompt from here.

Each agent gets: the model's top-ranked candidates for its scope (name, club,
price, model xP), then researches live news to catch what the stats model
structurally can't see, and returns a structured table (player, status flag,
rationale, source). The orchestrator consolidates all of it plus
`docs/player_judgments.md` into the final squad.

---

## 1. GK agent

_Last tweaked by Neil -- shifted from a mostly news-driven agent to a
mostly stats-driven one. Rationale below each change._

**Update: GK and DEF now both look 4 gameweeks ahead, not just the next
match.** `models/fixture_run.py` computes each team's expected clean-sheet
run over a window (default GW1-4, configurable), combines it with each
player's own defensive output (clean sheets for keepers, tackles +
clearances/blocks/interceptions for defenders, from real 2025/26 data), and
gates the whole thing on start probability -- a good fixture run behind a
player who won't actually play is worthless, so anyone below a 40% start
chance is dropped before ranking, not after. Every row also reports whether
its start probability is backed by real data or just the flat 65% guess
(`p_start_grounded`), so a shaky pick can't hide inside a good-looking
score. Real run for GW1-4 right now: Liverpool, Chelsea, Man Utd, Arsenal,
and Man City have the kindest defensive fixture runs; Fulham, Newcastle,
Hull, and Ipswich have the toughest.

**Update: the "first-choice via last season's starts" check is now built
into the model itself**, not just the agent. `xp_model.py` pulls each
player's real 2025/26 starts/minutes (from `data/raw/historical_2025_26/`,
joined by FPL's stable player `code`) and uses that as the start-probability
prior pre-season, instead of a flat 65% guess for everyone. 458 of 573
players (80%) have real history to use this way; this is what correctly
buries Meslier's start probability before any news search even runs. The
agent's job narrows to what the season stat alone can't catch: summer
transfers (a great last-season start rate at his old club doesn't mean
he's first-choice at a new one), and the ~20% of players with no PL history
at all (promoted teams, brand new arrivals), where research is still the
only signal available.

**What it's handed:** the model's top ~16 goalkeeper candidates by expected
points.

**What it's asked to figure out:**

- **First-choice status: primarily last season's PL starts while fit,** not
  news search. Count of starts (excluding spells out injured) is the
  primary signal for who's #1 at a club. News search is a secondary check
  only for things last season's data can't show -- a summer transfer moving
  a keeper to a new club (this is exactly the Meslier-to-Arsenal case: his
  Leeds start count last season doesn't tell you he's now Arsenal's
  third-choice, so a transfer check is still needed on top of the stat).

- **Injury monitoring: dropped.** Neil's call: keeper injuries are rare and
  usually unpredictable freak incidents, not worth researching each week.
  Note this is about the *agent's* research effort, not the model itself --
  `xp_model.py` still zeroes out anyone the live FPL data flags as
  officially injured/suspended (`status`/`chance_of_playing_next_round`),
  that's a data-layer check that stays regardless.

- **Penalty-save duty: de-emphasized.** Previously a factor; now a minor
  one. Replaced with a team-level signal instead: **penalties conceded by
  the team last season.** A team that gives away a lot of penalties makes
  clean sheets harder to come by for their keeper, regardless of who's
  taking them.

- **Defense strength in front of him -- now several explicit team-level
  stats from last season**, not just a general "how good is their
  defense" read:
  - Clean sheets (last season).
  - Tackles per game.
  - Clearances per game.
  - **Possession %** -- Neil's addition: teams that control the ball more
    concede less, so this is a defensive proxy as much as an attacking one.

- **Points-per-million, specifically for evaluating the budget/bench
  keeper pick.** The top pick is chosen on starts + defensive quality
  above; the *second*, cheap keeper (who exists purely to free up budget
  elsewhere) should be screened by last season's FPL value (points per
  £m), not by upside.

**Output:** table of player / club / start status (nailed, rotation risk,
backup-don't-pick) / flag (upgrade, neutral, downgrade, avoid) / rationale
with source.

**Data note:** this shifts the agent from "mostly web research" to "mostly
structured stats lookup, with web research filling specific gaps (transfers,
new-manager changes)." The live pipeline (`data/fpl_client.py`) doesn't
currently carry last-season team-level tackles/clearances/possession/
penalties-conceded, or points-per-million -- these exist in the historical
archive already downloaded for backtesting (`data/raw/historical_2025_26/`)
and would need to be pulled into a proper reference table before this agent
can actually use them systematically instead of ad hoc per-run web search.

---

## 2. DEF agent

**Update: same 4-gameweek fixture-run change as GK (see section 1)** --
`models/fixture_run.py`'s `defensive_shortlist()` is the candidate-generation
step feeding this agent now, not a single-gameweek xP sort. Combines each
defender's team's GW1-4 clean-sheet run with his own tackle/clearance
history, gated on a real start-probability check.

**What it's handed:** top ~16 defender candidates by expected points.

**What it's asked to figure out:**
- Confirmed starting-XI status.
- Injury/fitness doubts.
- Whether the player is a genuine attacking threat (set pieces, forward
  runs) -- matters for FPL points beyond clean sheets, and the model
  doesn't have penalty/set-piece duty data at all yet.
- How new signings are settling in, or whether a new manager's system
  favors them.
- Sanity-checks surprising club assignments (e.g. a player shown at a new
  club) against real transfer news rather than assuming the data is stale.

**Output:** same table format as GK.

---

## 3. MID agent

**What it's handed:** top ~16 midfielder candidates by expected points.

**What it's asked to figure out:**
- Confirmed starting-XI role.
- Penalty and set-piece (corners/free-kicks) duty specifically -- this is
  flagged as a major swing factor for midfielders that the model has no
  visibility into on its own.
- Injury/fitness doubts.
- How well a recently-transferred player is fitting a new team/system.
- Whether premium picks still hold penalty duty at their club (duty changes
  hands more than people expect).

**Output:** table with an extra column versus GK/DEF: penalty/set-piece duty
(yes/no/shared).

---

## 4. FWD agent

**What it's handed:** top ~16 forward candidates by expected points, plus
which ones are already flagged doubtful/injured by the live data.

**What it's asked to figure out:**
- Confirmed starting-XI status.
- Penalty duty (the single biggest scoring factor for forwards the model
  can't see).
- Latest update on any flagged injuries -- fit for this gameweek or not.
- How well new signings are settling into the attack.
- Sanity-checks surprising club assignments against real transfer news.

**Output:** same format as MID (includes penalty duty column).

---

## 5. Fixtures agent

**What it's handed:** the gameweek's full fixture list.

**What it's asked to figure out:**
- Full fixture list with kickoff times.
- Per fixture: anything relevant to FPL decisions -- a promoted team
  facing a weak/strong opponent, a new manager who might set up
  differently than expected, key absences specific to that match, which
  fixtures look like good clean-sheet/attacking opportunities vs. ones to
  avoid targeting.
- Explicit low-confidence flags where preseason form/transfer hype is the
  only signal available (which it usually is, early in a season).

**Output:** full fixture table with home/away/toss-up lean and notes, plus a
short "watch list" of the 3-5 fixtures most worth targeting or avoiding.

---

## 6. Chip strategy agent

**What it's handed:** just the gameweek number and season context (nothing
position-specific -- it doesn't see player candidates at all).

**What it's asked to figure out:**
- How many of each chip (Wildcard, Free Hit, Bench Boost, Triple Captain)
  are available this season, and any season-specific rule changes.
- Whether anything about this gameweek specifically (blank/double
  gameweek, squad crisis) warrants deviating from "hold chips."
- A direct recommendation for this gameweek.

**Output:** short brief, not a table -- chip availability, any reason to
deviate from holding, and a recommendation.

---

## Known gaps / things worth tweaking

- None of the agents currently see `docs/football_domain_knowledge.md` or
  each other's output -- they research independently and the orchestrator
  is the only place synthesis happens. Worth discussing whether e.g. the
  fixtures agent's output should feed into the position agents directly
  instead of being combined only at the end.
- The DEF/MID/FWD agents rely on a text list of names/prices typed into the
  prompt by whoever dispatches them, not a live data file -- that's a
  manual step that doesn't scale well to weekly automation (build step 10).
- "Downgrade" is a soft, single-step penalty in the current override system
  (`apply_manual_adjustments`). The GW1 backtest fix surfaced a real case
  (Welbeck) where a "downgrade" wasn't strong enough to survive better
  numbers elsewhere in the squad -- worth discussing whether agents need a
  stronger "bench-only, don't consider" flag distinct from "downgrade."
