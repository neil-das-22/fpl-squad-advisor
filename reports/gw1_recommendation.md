# GW1 Squad Recommendation — 2026/27 Season

Deadline: Friday 21 August 2026, 17:30 BST. Opening fixture: Arsenal vs Coventry City, Fri 21 Aug, 20:00 BST.

Built from: live 2026/27 preseason data (573 players, 20 clubs) via the FPL API, a statistical expected-points model backtested against the full real 2025/26 season, research from six specialist agents (goalkeepers, defenders, midfielders, forwards, fixtures, chip strategy), and Neil's own judgment calls from `docs/player_judgments.md`.

**Updated after backtesting** (see `backtest/results_2025_26.md`): the model had a bug where it couldn't tell "no data yet" apart from "confirmed non-player," and it was guessing at defensive-contribution points instead of using each player's own tackle/interception history, which the live data pipeline now pulls in. Both are fixed; this squad reflects the corrected model. One side effect worth knowing: Chelsea's Welbeck briefly appeared in the optimized squad once the numbers shifted, despite our own research flagging him as a backup, not a starter — excluded him explicitly rather than let a soft score adjustment get overridden by better numbers elsewhere.

## Recommended 15-man squad

| Player | Club | Pos | Price | xP |
|---|---|---|---|---|
| Raya | Arsenal | GKP | £6.0m | 4.73 |
| Dubravka | Spurs | GKP | £4.0m | 2.72 |
| Lacroix | Chelsea | DEF | £6.0m | 5.76 |
| Virgil van Dijk | Liverpool | DEF | £6.5m | 5.75 |
| N.Williams | Nott'm Forest | DEF | £5.0m | 5.11 |
| Shaw | Man Utd | DEF | £4.5m | 4.68 |
| Calafiori | Arsenal | DEF | £5.5m | 4.66 |
| Mbeumo | Man Utd | MID | £8.0m | 6.51 |
| Enzo Fernández | Chelsea | MID | £7.0m | 6.30 |
| Gakpo | Liverpool | MID | £7.0m | 5.91 |
| Amad | Man Utd | MID | £6.0m | 4.90 |
| Zubimendi | Arsenal | MID | £5.5m | 3.83 |
| Haaland | Man City | FWD | £15.5m | 7.35 |
| João Pedro | Chelsea | FWD | £7.5m | 5.91 |
| Igor Jesus | Nott'm Forest | FWD | £6.0m | 4.17 |

**Cost: £100.0m / £100.0m** (no money left in the bank)

## Starting XI — 4-4-2

**GK:** Raya
**DEF:** Lacroix, Virgil van Dijk, N.Williams, Shaw
**MID:** Mbeumo, Enzo Fernández, Gakpo, Amad
**FWD:** Haaland, João Pedro

**Captain: Haaland** (7.35 xP, doubled to 14.70)
**Vice-captain: Mbeumo**

**Bench (auto-sub order):** 1. Calafiori (DEF) 2. Igor Jesus (FWD) 3. Zubimendi (MID) 4. Dubravka (GKP)

XI expected points before captaincy: 62.92. With captaincy: **70.27.**

## The one judgment call worth knowing about: Haaland

Your `player_judgments.md` flags Haaland as a must-have. Left purely to the optimizer, it would drop him — his £15.5m price tag doesn't clear its bar relative to cheaper forwards, and the pure-math squad scores 81.98 xP versus 78.29 xP with him forced in. That's a real, quantified cost: **3.69 expected points this gameweek** to hold Haaland (this went up slightly from the original 2.85 estimate now that the rest of the squad's numbers are more accurate).

I built it forcing him in, on the view that your judgment on a proven, high-floor penalty-taking striker should override a single-gameweek optimizer number, especially this early in a season with thin data. But this is exactly the kind of call that should be yours, not mine by default — if you'd rather bank the 3.69 points and go with the pure-math squad instead, say so and I'll rerun it.

## Why these players, per position (from the research agents)

**GK:** Raya is Arsenal's undisputed #1, and the season opener is at home against newly-promoted Coventry — about as clean-sheet-friendly as GW1 fixtures get. Dubravka is a pure budget bench-warmer to free up funds for Haaland.

**DEF:** Lacroix (Chelsea's £52m marquee signing, straight into the first-choice CB pairing) and Virgil van Dijk (Liverpool captain, still the clearest defensive floor in the league) anchor this line. N.Williams and Shaw are established starters with attacking upside. Calafiori (Arsenal, strong preseason form) is the bench option most likely to earn a swap-in. Gabriel (Arsenal) and Dalot/Gusto were downgraded on injury and rotation-risk grounds respectively, despite decent model scores.

**MID:** Mbeumo, Enzo Fernández, and Gakpo all carry either penalty/set-piece duty or strong underlying numbers and confirmed starting roles. Enzo's inclusion carries some risk — he was missing from Chelsea's most recent predicted XI (Lavia/Caicedo may be preferred in the pivot) — worth a last look before the deadline. Amad and Zubimendi round out the squad as value picks in settled attacking/midfield roles.

**FWD:** Haaland (nailed, #1 penalties) per your judgment call above. João Pedro is Chelsea's confirmed starting striker. Igor Jesus (Nottingham Forest) is in-form and a genuine starting contender per preseason reports, and sits on the bench as the most likely auto-sub.

**Notably excluded** despite reasonable model scores: Meslier, Heaton, Davies, Pecsi, Penders (all confirmed backups, not starters), Chalobah (reportedly exiting for Como), Amass and Ramsay (buried in their depth charts), N.Jackson, Obi, and Welbeck (unlikely to feature in a reshaped Chelsea/Man Utd forward line).

## Fixtures watch (GW1)

Best fixtures on the board: **Arsenal vs Coventry** and **Man City vs Bournemouth** — both strong home favorites against weaker opposition, low-risk for clean sheets and attacking returns. **Hull vs Man Utd** favors United but Hull's own assets are a low-confidence unknown (no top-flight data). **Newcastle vs Liverpool** is the marquee toss-up of the round — avoid banking on a clean sheet from either side. Everything else on the card is close to a coin flip; preseason form and transfer hype are weak signals this early, so treat any "favorite" tag as a base rate, not a certainty.

## Chip strategy

**Hold everything.** Wildcard and Free Hit are mechanically blocked in GW1 anyway (you can't wildcard a squad you haven't picked). Bench Boost and Triple Captain are technically usable from GW1, but there's no blank/double gameweek or information edge this early to justify it — building this squad is already effectively a "free wildcard." Save all four chips for when a real signal (fixture swing, DGW, squad crisis) shows up later in the season.

## Confidence notes

This is built on real live 2026/27 preseason data, but it's still preseason — there's no actual gameweek data yet to validate against for this specific season. What we do have now is a model backtested against the full real 2025/26 season (see `backtest/results_2025_26.md`): it beats naive baselines on the players worth picking from, though it still trails FPL's own official projections, which have access to confirmed lineups we don't. The research agents caught several things the numbers alone would have missed (Meslier's real move to Arsenal as third-choice keeper, Guéhi and Lacroix's genuine transfers, Gyökeres's new penalty duty), which is the point of running this hybrid instead of pure optimization. A few research threads (Alisson's exact fitness status, some Chelsea/Arsenal team-news specifics) hit a search rate limit and are flagged as lower-confidence in the underlying agent reports — worth a final news check before the Friday deadline.
