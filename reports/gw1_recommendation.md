# GW1 Squad Recommendation — 2026/27 Season

Deadline: Friday 21 August 2026, 17:30 BST. Opening fixture: Arsenal vs Coventry City, Fri 21 Aug, 20:00 BST.

Built from: live 2026/27 preseason data (573 players, 20 clubs) via the FPL API, a statistical expected-points model, research from six specialist agents (goalkeepers, defenders, midfielders, forwards, fixtures, chip strategy), and Neil's own judgment calls from `docs/player_judgments.md`.

## Recommended 15-man squad

| Player | Club | Pos | Price | xP |
|---|---|---|---|---|
| Raya | Arsenal | GKP | £6.0m | 4.73 |
| Dubravka | Spurs | GKP | £4.0m | 2.72 |
| Lacroix | Chelsea | DEF | £6.0m | 5.20 |
| N.Williams | Nott'm Forest | DEF | £5.0m | 5.03 |
| Shaw | Man Utd | DEF | £4.5m | 4.67 |
| Lucky | Liverpool | DEF | £4.0m | 3.51 |
| O.Richards | Nott'm Forest | DEF | £4.0m | 3.42 |
| Mbeumo | Man Utd | MID | £8.0m | 6.67 |
| Gibbs-White | Nott'm Forest | MID | £8.0m | 6.42 |
| Enzo Fernández | Chelsea | MID | £7.0m | 6.39 |
| Gakpo | Liverpool | MID | £7.0m | 6.08 |
| Amad | Man Utd | MID | £6.0m | 4.94 |
| Haaland | Man City | FWD | £15.5m | 7.37 |
| João Pedro | Chelsea | FWD | £7.5m | 5.92 |
| Gyökeres | Arsenal | FWD | £7.5m | 5.55 |

**Cost: £100.0m / £100.0m** (no money left in the bank)

## Starting XI — 3-4-3

**GK:** Raya
**DEF:** Lacroix, N.Williams, Shaw
**MID:** Mbeumo, Gibbs-White, Enzo Fernández, Gakpo
**FWD:** Haaland, João Pedro, Gyökeres

**Captain: Haaland** (7.37 xP, doubled to 14.74)
**Vice-captain: Mbeumo**

**Bench (auto-sub order):** 1. Amad (MID) 2. Lucky (DEF) 3. O.Richards (DEF) 4. Dubravka (GKP)

XI expected points before captaincy: 64.02. With captaincy: **71.39.**

## The one judgment call worth knowing about: Haaland

Your `player_judgments.md` flags Haaland as a must-have. Left purely to the optimizer, it would drop him — his £15.5m price tag doesn't clear its bar relative to cheaper forwards, and the pure-math squad scores 81.46 xP versus 78.61 xP with him forced in. That's a real, quantified cost: **2.85 expected points this gameweek** to hold Haaland.

I built it forcing him in, on the view that your judgment on a proven, high-floor penalty-taking striker should override a single-gameweek optimizer number, especially this early in a season with thin data. But this is exactly the kind of call that should be yours, not mine by default — if you'd rather bank the 2.85 points and go with the pure-math squad instead (swaps out Haaland and Dubravka for a stronger XI elsewhere), say so and I'll rerun it.

## Why these players, per position (from the research agents)

**GK:** Raya is Arsenal's undisputed #1, and the season opener is at home against newly-promoted Coventry — about as clean-sheet-friendly as GW1 fixtures get. Dubravka is a pure budget bench-warmer to free up funds for Haaland.

**DEF:** Lacroix (Chelsea's £52m marquee signing, straight into the first-choice CB pairing), N.Williams (Nottingham Forest's attacking wing-back, retained under new boss Glasner), and Shaw (established Man Utd starter) were the strongest combination of nailed-on and attacking upside within budget. Gabriel (Arsenal) and Dalot/Gusto were downgraded on injury and rotation-risk grounds respectively, despite decent model scores.

**MID:** Mbeumo, Gibbs-White, Enzo Fernández, and Gakpo all carry either penalty/set-piece duty or strong underlying numbers and confirmed starting roles. Enzo's inclusion carries some risk — he was missing from Chelsea's most recent predicted XI (Lavia/Caicedo may be preferred in the pivot) — worth a last look before the deadline.

**FWD:** Haaland (nailed, #1 penalties) per your judgment call above. Gyökeres is a genuine upgrade over his model score — he's reportedly now Arsenal's #1 penalty taker ahead of Saka, which the historical-data model couldn't have known. João Pedro is Chelsea's confirmed starting striker.

**Notably excluded** despite reasonable model scores: Meslier, Heaton, Davies, Pecsi, Penders (all confirmed backups, not starters), Chalobah (reportedly exiting for Como), Amass and Ramsay (buried in their depth charts), N.Jackson and Obi (unlikely to feature in a reshaped Chelsea/Man Utd forward line).

## Fixtures watch (GW1)

Best fixtures on the board: **Arsenal vs Coventry** and **Man City vs Bournemouth** — both strong home favorites against weaker opposition, low-risk for clean sheets and attacking returns. **Hull vs Man Utd** favors United but Hull's own assets are a low-confidence unknown (no top-flight data). **Newcastle vs Liverpool** is the marquee toss-up of the round — avoid banking on a clean sheet from either side. Everything else on the card is close to a coin flip; preseason form and transfer hype are weak signals this early, so treat any "favorite" tag as a base rate, not a certainty.

## Chip strategy

**Hold everything.** Wildcard and Free Hit are mechanically blocked in GW1 anyway (you can't wildcard a squad you haven't picked). Bench Boost and Triple Captain are technically usable from GW1, but there's no blank/double gameweek or information edge this early to justify it — building this squad is already effectively a "free wildcard." Save all four chips for when a real signal (fixture swing, DGW, squad crisis) shows up later in the season.

## Confidence notes

This is built on real live 2026/27 preseason data, but it's still preseason — there's no actual gameweek data yet to validate the model against. The research agents caught several things the numbers alone would have missed (Meslier's real move to Arsenal as third-choice keeper, Guéhi and Lacroix's genuine transfers, Gyökeres's new penalty duty), which is the point of running this hybrid instead of pure optimization. A few research threads (Alisson's exact fitness status, some Chelsea/Arsenal team-news specifics) hit a search rate limit and are flagged as lower-confidence in the underlying agent reports — worth a final news check before the Friday deadline.
