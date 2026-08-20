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

**Update: rebuilt around two different stat profiles instead of one.**
`models/fixture_run.py`'s new `midfielder_shortlist()` is the
candidate-generation step now, replacing a flat top-16-by-xP sort.
Rationale below.

Neil's core point: a defensive midfielder and a winger both get classified
"MID" by FPL, but they score points completely differently, so judging
both against the same criteria structurally punishes whichever one the
criteria don't favour. So instead of one score, every candidate gets
**both**:
- **`attacking_score`** -- shrunk per-90 blend of prior-season goals,
  assists, expected goal involvement (xG+xA), and creativity.
- **`defensive_score`** -- shrunk per-90 tackles + clearances/blocks/
  interceptions (the same CBIT count the model's own MID DefCon term
  uses).

`profile_score = max(attacking_z, defensive_z)` -- a player is judged on
whichever half of the job he actually does. This is also the mechanism
that satisfies **"wingers should be in high regard as midfielders"**:
there's no "is this a winger" field anywhere in FPL's data to key off of,
so rather than inventing a fake classification, a genuine winger simply
racks up a high attacking_score and wins on that axis. It's implicit, not
a formal winger tag -- worth knowing if a specific player's classification
ever looks off.

**Data honesty note:** FPL's public API has no pass-completion counts and
no standalone "big chances created" stat -- that's Opta-tier data the API
doesn't publish. `creativity` (an ICT sub-index built from chance-creation
events) is used as the closest real substitute everywhere "chance
creation" is asked for. This is a documented proxy, not silently treated
as the real thing.

**Other criteria, each its own visible column rather than folded invisibly
into one number** (per Neil's DEF-agent feedback that he wants to see the
actual stats, not just a score):
- **Discipline record now counts against a player.** `discipline_score` =
  shrunk cards-per-90 (yellow + red, weighted 1:3), subtracted from
  run_score. A high-output player with a poor card record still ranks
  lower than an equally productive but cleaner one.
- **Set-piece duty is now real, structured data, not just research.**
  `penalties_order` / `corners_and_indirect_freekicks_order` /
  `direct_freekicks_order` are published directly by the FPL API (1 =
  primary taker) -- see `data/fpl_client.py`. Previously the agent had to
  find this via web search every week; now it's pulled straight from the
  data and only needs confirming, not discovering.
- **Team attacking context** (Neil: "the attackers on the same team help
  the midfielders get assists/goals"). Rather than picking an arbitrary
  "front three" to aggregate (no clean definition exists in the data),
  this uses FPL's own team-level attack-strength rating
  (`team_attack_strength()`, from `strength_attack_home/away`) -- a
  real, club-wide measure of how dangerous the team's attackers
  collectively are. Flagged clearly as a team-level proxy, not a
  hand-picked sum of specific teammates' stats.
- **4-gameweek fixture run, attacking framing.** Same machinery as GK/DEF
  (`models/fixture_run.py`), extended: `mean_attack_multiplier` over the
  window (favourable = facing weak defences, the mirror image of GK/DEF's
  clean-sheet run).

**Two self-caught bugs while building this, both fixed and covered by new
tests before this went anywhere near a squad recommendation:**
1. `team_attack_strength` initially came back as a flat 0.0 for all 20
   clubs -- the live pre-season data has `strength_attack_home/away` set
   to 0 for everyone (FPL hasn't published attack/defence sub-ratings this
   early), which the rest of the model already knows to work around
   (`xp_model._strength()` falls back to `strength_overall_home`, which
   *is* populated) but this new function initially read the raw column
   directly and missed that fallback. Fixed by reusing the existing
   fallback logic instead of a second, silently-broken read path.
2. A player with literally no prior-season row (no Premier League history
   at all -- promoted-team debuts, brand-new arrivals) had every input to
   both scores computed off flat priors, and the MID DefCon prior (9.3
   CBIT/90, calibrated to the *population average*) sits ABOVE most real
   attacking stars' actual, structurally lower defensive output. Left
   unguarded, `profile_score`'s max() let a total unknown's "average"
   defensive guess outrank proven attacking players purely because the
   guess landed generously -- the same guess-dressed-as-evidence problem
   `p_start_grounded` exists to catch on the start-probability side. Fixed
   with a second, separate gate: anyone with no real prior-season minutes
   is excluded from the ranked list entirely (43 of the live MID pool
   right now) rather than scored on a guess, and reported back separately
   (`result.attrs["excluded_no_prior_data"]`) so they're not silently lost
   either -- same "flat fallback is honest, not hidden" principle as the
   ~20%-no-history group documented elsewhere in this file.

**Real numbers right now (GW1-4 window):** top of the list is Bruno
Fernandes (Man Utd, penalties + direct free-kicks, £12.0m), then Cherki
(Man City, direct free-kicks), Saka (Arsenal, penalties), Doku, Wirtz,
Mbeumo, Foden, Palmer, Szoboszlai, Enzo Fernández, Rice -- all ranking on
real prior-season output plus a real team-attack and fixture-run context,
not a flat guess.

**What it's still asked to figure out (the part stats can't cover):**
- Confirmed starting-XI role -- especially for the 43 MIDs excluded above
  for having no prior-season data; this is the one research pass that
  still matters for them.
- Injury/fitness doubts.
- How well a recently-transferred player is fitting a new team/system.
- Whether premium picks still hold penalty/set-piece duty this season --
  the live data reflects last season's assignment until proven otherwise
  (same carryover behaviour documented throughout this project), so a
  summer change of duty needs a news check, not just a data read.

**Output:** table with attacking_score, defensive_score, profile_score,
discipline_score, set_piece_duty, mean_attack_multiplier,
team_attack_strength, run_score -- plus a separate short list of anyone
excluded for having no prior-season data at all, flagged for research
rather than silently dropped.

---

## 4. FWD agent

**Update: rebuilt around service quality, not just a striker's own
numbers in isolation.** `models/fixture_run.py`'s new `forward_shortlist()`
is the candidate-generation step now, replacing a flat top-16-by-xP sort.
Two of Neil's specific requests didn't exist anywhere in this project
before this rebuild:

- **`team_creative_supply`** -- "I want to consider the attacking
  midfielder and wingers in their team, and their chances created,
  passing, assist stats." Rather than picking an arbitrary "front three"
  of teammates to aggregate, this sums the shrunk per-90 supply output
  (assists + xA + creativity) of the club's own top-3 creative MIDs
  (`team_creative_supply()`) -- a striker's actual supply line,
  quantified from real data, not a hand-picked list.
- **`team_possession_pct`** -- "possession of the team is important for
  both midfielders and attackers." Neither the FPL API nor the vaastav
  archive publishes team possession at all (checked directly, every
  column header in every file this project uses -- it isn't there).
  Sourced live from Sofascore's team-statistics API for the real 2025/26
  season (fbref blocked the same pull behind bot detection;
  premierleague.com's stats pages returned dead links -- Sofascore's own
  API worked cleanly via a Chrome fetch). Covers 17 of the current
  season's 20 clubs; this season's 3 promoted teams (Coventry, Ipswich,
  Hull) played Championship football last season and have no top-flight
  possession data to pull -- left NaN, not guessed. Also retroactively
  added to `midfielder_shortlist()`, since Neil's ask covered both
  positions.
- **Bonus find while sourcing possession:** Sofascore's team-statistics
  endpoint also has `bigChancesCreated` and real pass-completion
  percentages at the team level -- the exact two stats flagged as
  "not available anywhere" during the MID rebuild. Not wired in yet
  (this was scoped to possession specifically), but worth revisiting if
  team-level chance-creation/passing ever becomes worth adding to MID or
  the fixtures agent.

**What it still computes, same treatment as MID/DEF:**
- `attacking_score` -- shrunk per-90 goals + assists, xG + xA, and threat
  (ICT's goal-threat index, the forward-specific counterpart to MID's
  creativity proxy). No blended attacking/defensive profile here --
  unlike MID there's no real "defensive forward" archetype to account
  for, so this is the single core signal.
- `set_piece_duty` -- penalty order specifically, real structured data
  from `penalties_order` (see fpl_client.py), not research-agent guesswork.
- `mean_attack_multiplier` -- same 4-gameweek attacking fixture run as MID.
- `discipline_score` -- cards per 90, same negative-weight treatment as MID.
- Same two gates as MID: excluded below the start-probability bar, and
  separately excluded (not scored on a guess) if there's no real
  prior-season row at all -- reported back via
  `result.attrs["excluded_no_prior_data"]`.

**Real numbers right now (GW1-4 window):** Haaland is a clear #1 (penalty
duty, £15.5m, by far the strongest attacking score and team context), then
João Pedro, Gyökeres, Welbeck, Thiago, Watkins, Mateta, Richarlison,
Calvert-Lewin. Worth flagging: Welbeck clears the model's own
start-probability bar (p_start 0.68) here -- the GW1 report excluded him
based on the FWD research agent's news that Chelsea's attack looks
reshaped this summer, which is exactly the kind of thing this stats-only
tool structurally can't see. The two layers are meant to disagree
sometimes; that disagreement is the research agent's job to resolve, not
a bug in either one.

**What it's still asked to figure out (the part stats can't cover):**
- Confirmed starting-XI status -- especially for anyone excluded above for
  having no prior-season data.
- Latest update on any flagged injuries -- fit for this gameweek or not.
- How well new signings are settling into the attack.
- Sanity-checks surprising club assignments against real transfer news.
- Squad-depth/tactical reshuffles a stats-only model can't see (the
  Welbeck case above).

**Output:** table with attacking_score, mean_attack_multiplier,
team_attack_strength, team_possession_pct, team_creative_supply,
set_piece_duty, discipline_score, run_score -- plus a separate list of
anyone excluded for having no prior-season data at all.

---

## 5. Fixtures agent

**Update: squad selection itself now looks 3-4 gameweeks ahead, not just
the next match -- this moved from "agent describes fixtures" to "the
model optimizes across a window."** Per Neil: "I want the fixtures agent
to not just look at the upcoming fixture, I want it to look 3-4 gameweeks
out and make a team based on xp over the next 3-4 week fixtures, and
keeping transfers in mind."

**How:** `agents/orchestrator.py`'s new `build_multi_gw_xp_table(start_gw,
n_gw=4)` runs `xp_model.calculate_xp_for_gameweek()` once per gameweek in
the window and sums each player's `xp` across it (each week's blanks/
doubles handled individually, so a double gameweek correctly counts
double and a blank correctly counts zero). `run_gameweek_multi_gw()` feeds
that cumulative table through the exact same tested pipeline as the
single-gameweek version -- research overrides, exclude/must-include,
`pick_squad()`, `pick_starting_xi()` -- unchanged, since the optimizer
never cared whether its `xp` column came from one week or four.

**Why this is "keeping transfers in mind" for a from-scratch build:** FPL
gives one free transfer a week (rolling to a max of 5 -- confirmed live,
see sources below). A squad picked purely to maximise gameweek 1's
fixtures can force an immediate, costly change the moment that week
passes. Optimising for cumulative value across the window instead
produces a squad that holds up for the whole stretch, not just its
opener -- there's no existing squad to transfer FROM yet at GW1, so this
is the right lever for that specific case.

**Real result just run (GW1-4, same overrides/excludes as the GW1
report):** cumulative squad xP 273.48 over the window, in a 3-4-3
shape. Notably different from the single-gameweek GW1 report: Raya
(Arsenal) drops out of the squad entirely in favour of a cheaper
Petrović/Dubravka goalkeeper pairing (frees up ~£1.5m that the optimizer
puts into a stronger outfield XI over the 4-week window) -- Bruno-style
name checks aside, this is the optimizer reallocating budget toward
cumulative value rather than a single strong week 1 fixture. Worth a look
before trusting it outright -- it hasn't been through the position
research agents' news check the way the current GW1 report has.

**Known simplification, not yet corrected:** the research-agent override
system (`apply_manual_adjustments`, flat +/-0.5 xP upgrade/downgrade
steps) was calibrated for single-gameweek magnitudes. Applied to a ~4x
larger cumulative total, the same flat step is proportionally weaker --
it doesn't automatically scale with the window. Not fixed here (there's
no clean way to rescale it without guessing what "half a gameweek's edge"
should mean spread over a month) -- just flagged so a research call
looking like it "did less" on the multi-gameweek table isn't mistaken for
a bug.

**Once a squad actually exists (post-GW1), not yet exercised for real
but already wired for it:** the same cumulative table can be handed to
the existing, unmodified `squad_optimizer.optimize_transfers()` -- it
already computes `net_gain = xP(in) - xP(out) - hit_cost * max(0, n -
free_transfers)`, so pointing it at a 4-gameweek `xp` column instead of a
1-gameweek one makes every future transfer decision weigh the next month,
not just next week, with zero new optimizer code required.

**What it's still asked to figure out (the qualitative part a difficulty
rating and an xP sum can't cover):**
- Full fixture list with kickoff times.
- Per fixture: anything relevant to FPL decisions -- a promoted team
  facing a weak/strong opponent, a new manager who might set up
  differently than expected, key absences specific to that match, which
  fixtures look like good clean-sheet/attacking opportunities vs. ones to
  avoid targeting.
- Explicit low-confidence flags where preseason form/transfer hype is the
  only signal available (which it usually is, early in a season).
- Sanity-checking the multi-gameweek squad above -- it's built from the
  same raw model + research overrides as the GW1 report, not yet
  re-vetted by the position agents against this new window.

**Output:** full fixture table with home/away/toss-up lean and notes, plus a
short "watch list" of the 3-5 fixtures most worth targeting or avoiding.

**Free transfer rules for reference (confirmed live, Aug 2026):** 1 free
transfer per gameweek, rolling if unused up to a maximum bank of 5;
transfers beyond the free allowance cost -4 points each.
[Premier League: FPL 2026/27 changes](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627),
[full90fpl: FPL Transfers Explained](https://full90fpl.com/fpl-transfers-explained/)

---

## 6. Chip strategy agent

**Update: chip deadlines are now tracked as a hard fact, not left to the
agent's memory each week.** Per Neil: "I think the chips reset at some
point through the season, just make sure to use all chips before they
reset." Confirmed live: 2026/27 gives one Wildcard, one Free Hit, one
Bench Boost, and one Triple Captain for the FIRST half of the season
(GW1-19), and a completely fresh set of all four for the second half
(GW20-38). **Unused first-half chips are lost at the GW19 deadline (13:30
GMT, Sat 2 Jan) -- they do not carry over.** This is the one genuinely
deterministic fact in an otherwise judgement-heavy area, so it's now code
(`agents/chip_strategy.py`), not something re-derived from memory every
dispatch.

`chip_strategy.chip_status(gameweek, chips_used_this_half)` returns, for
every chip: whether it's been used this half, and an urgency level --
`hold` (plenty of runway), `plan_soon` (within 6 gameweeks of the
deadline -- start actively looking for a fixture to use it on), or
`urgent` (within 3 gameweeks -- use it or lose it). This is a genuine
use-it-or-lose-it deadline, not a soft suggestion, so the agent's
recommendation should escalate accordingly as a half's deadline
approaches, not just repeat "hold" by default the way it correctly did at
GW1.

**What it's handed:** the gameweek number, season context, and now
`chip_status()`'s output for the current half (which chips are used,
which are unused, and each unused chip's urgency).

**What it's asked to figure out:**
- Whether anything about THIS gameweek specifically (blank/double
  gameweek, squad crisis) warrants using a chip now, on top of the
  deadline-driven urgency above -- a double gameweek showing up while a
  chip sits at `plan_soon` is a good reason to act early rather than wait
  for `urgent` to force the decision.
- Once urgency hits `urgent`, an explicit call on which upcoming gameweek
  (within the remaining window) is the best of the available options --
  "hold" is no longer a valid recommendation once the deadline is that
  close, since holding past it means losing the chip outright.
- A direct recommendation for this gameweek.

**Output:** short brief, not a table -- chip availability + urgency (now
computed, not guessed), any reason to deviate from holding, and a
recommendation.

**What this does NOT do:** decide WHEN within a half to use a chip based
on football judgement (which gameweek's fixtures/blank/double actually
justify it) -- that qualitative call stays the agent's job. The code only
answers "how many gameweeks are left before an unused chip disappears."

Sources:
[Premier League: What's happening with FPL chips in 2026/27](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627),
[Premier League: How and when to use your chips in 2026/27 Fantasy](https://www.premierleague.com/en/news/4362085)

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
