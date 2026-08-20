# GW1 Squad Recommendation — 2026/27 Season

Deadline: Friday 21 August 2026, 17:30 BST. Opening fixture: Arsenal vs Coventry City, Fri 21 Aug, 20:00 BST.

Built from: live 2026/27 preseason data (573 players, 20 clubs) via the FPL API, a statistical expected-points model backtested against the full real 2025/26 season, real 2025/26 team possession/attack-context data sourced from Sofascore, and Neil's own judgment calls from `docs/player_judgments.md`.

**Updated since the last version.** The squad is now optimized for cumulative expected points across gameweeks 1-4, not just GW1 in isolation — a squad picked purely for its opening fixture can force an expensive change the moment that week passes, and only one free transfer exists per gameweek. This also fixed a real bug: the DEF research agent's downgrade on Van Dijk (missed preseason friendlies) was silently failing to apply, because FPL's own `web_name` for him is "Virgil," not "Van Dijk" — the override CSV never matched. Fixed and reflected below.

## Recommended 15-man squad

| Player | Club | Pos | Price | xP (cumulative, GW1-4) |
|---|---|---|---|---|
| Petrović | Bournemouth | GKP | £4.5m | 11.95 |
| Dubravka | Spurs | GKP | £4.0m | 11.55 |
| Virgil (van Dijk) | Liverpool | DEF | £6.5m | 21.58 |
| Lacroix | Chelsea | DEF | £6.0m | 19.03 |
| Guéhi | Man City | DEF | £6.0m | 16.97 |
| Shaw | Man Utd | DEF | £4.5m | 15.41 |
| Kayode | Brentford | DEF | £4.5m | 13.81 |
| Enzo Fernández | Chelsea | MID | £7.0m | 22.57 |
| Gakpo | Liverpool | MID | £7.0m | 20.70 |
| Szoboszlai | Liverpool | MID | £7.0m | 19.68 |
| Schade | Brentford | MID | £6.0m | 17.51 |
| Amad | Man Utd | MID | £6.0m | 15.22 |
| Haaland | Man City | FWD | £15.5m | 25.34 |
| Thiago | Brentford | FWD | £8.0m | 21.48 |
| João Pedro | Chelsea | FWD | £7.5m | 20.19 |

**Cost: £100.0m / £100.0m** (no money left in the bank) — **cumulative squad xP over GW1-4: 272.98**

## Starting XI — 3-4-3

**GK:** Petrović
**DEF:** Virgil (van Dijk), Lacroix, Guéhi
**MID:** Enzo Fernández, Gakpo, Szoboszlai, Schade
**FWD:** Haaland, Thiago, João Pedro

**Captain: Haaland**
**Vice-captain: Enzo Fernández**

**Bench (auto-sub order):** 1. Shaw (DEF) 2. Amad (MID) 3. Kayode (DEF) 4. Dubravka (GKP)

## The two judgment calls worth knowing about

### 1. Haaland — same call as before, cost has settled

Your `player_judgments.md` flags Haaland as a must-have. Left purely to the optimizer, it would drop him: the pure-math squad scores 277.64 cumulative xP versus 272.98 with him forced in. **Cost: 4.65 expected points over the 4-week window** (this has stayed close to flat since the model's last update — the underlying trade-off has settled, not just this gameweek's number).

Built forcing him in, same reasoning as before: your judgment on a proven, high-floor penalty-taking striker should override an optimizer number, especially with data still this thin. Still your call to reverse.

### 2. NEW — Raya vs. the cheap goalkeeper pairing

This is a genuinely new trade-off the multi-gameweek view surfaced. The model's own numbers actually rate Raya *higher* than Petrović individually — 13.67 cumulative xP vs. 11.95 — and a separate research tool that layers in real fixture-run and defensive data ranks him the clear #1 keeper in the league right now. But the optimizer doesn't pick goalkeepers in isolation: keeping Raya and a competent backup costs enough extra budget that it has to come from somewhere else in the squad, and the ILP found that reallocating that money elsewhere nets **more** total points than upgrading the keeper does.

Quantified: forcing Raya in (plus a viable second keeper) costs **4.67 expected points over GW1-4**, and cascades into weaker depth elsewhere (Guéhi and Kayode get swapped for cheaper, less-proven defenders to make the budget work). That's a real number, not a guess — worth weighing against how much you trust Raya's reputation over 4 weeks of thin data. I didn't force this one either way; it's presented as a call for you.

## Why the rest, and where the two research tools disagree

**DEF:** Virgil (van Dijk), Lacroix, and Guéhi anchor the line. Worth noting directly, since you were skeptical of Van Dijk earlier this project: the richer defensive research tool (real prior-season tackles/clearances + a genuine 4-gameweek fixture run, not just the optimizer's raw number) independently ranks him the **#2 defender in the league right now** behind only Lacroix, with a well-grounded 95% start probability. That's real data pushing back on the "too old, bad season" read, not the model just trusting his reputation — worth knowing even with the (now correctly-applied) fitness downgrade already factored in.

**MID:** Enzo, Gakpo, and Szoboszlai are strong picks in both the optimizer and the richer research tool. One real divergence worth flagging: the research tool (which weighs prior-season creativity, team attacking context, and set-piece duty more heavily) ranks Bruno Fernandes, Saka, Cherki, Doku, Wirtz, and Foden above everyone in this squad — but at their prices (Fernandes £12.0m, Saka £9.5m), they don't clear the budget bar once Haaland's £15.5m is accounted for. Enzo at £7.0m returns nearly as much raw model xP as Fernandes at £12.0m — that value gap is what the optimizer is actually optimizing for. Not a disagreement about who's good, just about who's worth it at the price.

**FWD:** Haaland per your call above. Thiago (Brentford) is a confirmed penalty taker with a strong underlying rate; João Pedro is Chelsea's starting striker. The forward research tool also rates Welbeck highly on pure output (he clears the model's own start-probability bar), which conflicts with his exclusion below — flagged as an open disagreement, not resolved here, since it hinges on squad-depth news the stats layer can't see.

**Notably excluded:** the GK backups (Meslier, Heaton, Davies, Pecsi, Penders — all confirmed non-starters by their actual 2025/26 records), DEF players buried in their depth charts (Chalobah, Amass, Ramsay), and FWD players unlikely to feature in a reshaped attack (N.Jackson, Obi, Welbeck — see the note above on this last one).

## Fixtures — now a 4-gameweek view, not just GW1

Squad selection already factors in the next 4 gameweeks' fixture difficulty for every player (not shown per-fixture here to keep this report focused on the squad; the underlying tool — `models/fixture_run.py` — can produce the full gameweek-by-gameweek breakdown for any player or team on request). Best individual GW1 fixtures on the board: **Arsenal vs Coventry** and **Man City vs Bournemouth**, both strong home favorites against weaker opposition. **Newcastle vs Liverpool** is the marquee toss-up of the round.

## Chip strategy

**Hold everything.** Wildcard and Free Hit are mechanically blocked in GW1 (you can't wildcard a squad you haven't picked). Bench Boost and Triple Captain are technically usable but there's no signal yet to justify burning one.

Worth knowing now, not later: **all four chips (Wildcard, Free Hit, Bench Boost, Triple Captain) reset at the Gameweek 19 deadline (13:30 GMT, Saturday 2 January)** — unused first-half chips are lost, not carried over, and a fresh set of all four unlocks for the second half. That's roughly 18 gameweeks of runway from here, plenty for now, but worth tracking as the deadline approaches rather than discovering it late. `agents/chip_strategy.py` now tracks this deadline directly and will flag urgency (plan-soon at 6 gameweeks out, urgent at 3) once we're closer.

## Confidence notes

The model is grounded in each player's actual 2025/26 Premier League record wherever one exists (80% of the pool), scored across a genuine 4-gameweek window rather than a single week, and cross-checked against a second, independently-built research layer (fixture-run + team context + real possession/creativity data) that agreed on most picks and surfaced two real, quantified trade-offs (Raya, and the Fernandes/Saka value gap) rather than silently picking a side. The weakest links: the ~20% of the pool with no PL history at all (promoted-team players, brand new arrivals — the flat prior is the honest answer, not a bug), the Welbeck disagreement noted above, and the fact this squad hasn't been through a fresh live news pass since the last research-agent dispatch — worth a final skim before Friday.
