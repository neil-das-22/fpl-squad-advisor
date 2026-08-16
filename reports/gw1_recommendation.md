# GW1 Squad Recommendation — 2026/27 Season

Deadline: Friday 21 August 2026, 17:30 BST. Opening fixture: Arsenal vs Coventry City, Fri 21 Aug, 20:00 BST.

Built from: live 2026/27 preseason data (573 players, 20 clubs) via the FPL API, a statistical expected-points model backtested against the full real 2025/26 season, research from six specialist agents (goalkeepers, defenders, midfielders, forwards, fixtures, chip strategy), and Neil's own judgment calls from `docs/player_judgments.md`.

**Updated twice since the first version.** First pass fixed two backtest-discovered bugs: the model couldn't tell "no data yet" apart from "confirmed non-player," and it was guessing at defensive-contribution points instead of using real tackle/interception history. Second pass closed a bigger gap: pre-season, the live FPL data still carries last season's numbers, and the model was accidentally treating that as if it were 37 gameweeks of the *new* season already played. It now correctly recognizes the new season hasn't started, and instead uses each player's actual 2025/26 season (starts, minutes, defensive stats) as the prior for who's likely to start GW1 — closing the exact gap the GK/DEF/MID/FWD research agents exist to patch over with manual news search. 458 of 573 players (80%) matched to real 2025/26 history this way; the rest (mostly players promoted-team squads or brand new to the league) still use the flat fallback, appropriately, since we genuinely have no Premier League data on them.

## Recommended 15-man squad

| Player | Club | Pos | Price | xP |
|---|---|---|---|---|
| Raya | Arsenal | GKP | £6.0m | 4.56 |
| Palmer | Ipswich Town | GKP | £4.0m | 2.71 |
| Virgil van Dijk | Liverpool | DEF | £6.5m | 5.61 |
| Lacroix | Chelsea | DEF | £6.0m | 5.57 |
| N.Williams | Nott'm Forest | DEF | £5.0m | 4.92 |
| Shaw | Man Utd | DEF | £4.5m | 4.57 |
| Bindon | Nott'm Forest | DEF | £4.0m | 3.61 |
| Mbeumo | Man Utd | MID | £8.0m | 6.35 |
| Enzo Fernández | Chelsea | MID | £7.0m | 6.05 |
| Gakpo | Liverpool | MID | £7.0m | 5.76 |
| Szoboszlai | Liverpool | MID | £7.0m | 5.46 |
| Amad | Man Utd | MID | £6.0m | 4.86 |
| Haaland | Man City | FWD | £15.5m | 7.12 |
| João Pedro | Chelsea | FWD | £7.5m | 5.77 |
| Igor Jesus | Nott'm Forest | FWD | £6.0m | 4.12 |

**Cost: £100.0m / £100.0m** (no money left in the bank)

Note: "Palmer" here is an Ipswich Town goalkeeper, not Cole Palmer of Chelsea — same surname, different player, different position. Worth double-checking on the official site before confirming, since a name collision like this is exactly the kind of thing worth a human glance.

## Starting XI — 3-5-2

**GK:** Raya
**DEF:** Virgil van Dijk, Lacroix, N.Williams
**MID:** Mbeumo, Enzo Fernández, Gakpo, Szoboszlai, Amad
**FWD:** Haaland, João Pedro

**Captain: Haaland**
**Vice-captain: Mbeumo**

**Bench (auto-sub order):** 1. Shaw (DEF) 2. Igor Jesus (FWD) 3. Bindon (DEF) 4. Palmer (GKP)

## The one judgment call worth knowing about: Haaland

Your `player_judgments.md` flags Haaland as a must-have. Left purely to the optimizer, it would drop him — his £15.5m price tag doesn't clear its bar relative to cheaper forwards, and the pure-math squad scores 79.66 xP versus 77.04 xP with him forced in. That's a real, quantified cost: **2.62 expected points this gameweek** to hold Haaland (this number has moved around a bit as the model's gotten more accurate — it was 2.85, then 3.69, now 2.62 — which is really just the noise of a single-gameweek estimate settling down as the underlying data improves, not the underlying trade-off changing character).

I built it forcing him in, on the view that your judgment on a proven, high-floor penalty-taking striker should override a single-gameweek optimizer number, especially this early in a season with thin data. Still your call to reverse if you'd rather bank the points.

## Why these players, per position (from the research agents + real last-season data)

**GK:** Raya is Arsenal's undisputed #1 — both the research agent's news search and his own 2025/26 record (37 of 38 possible starts) agree. Palmer (Ipswich) is a pure budget bench-warmer to free up funds for Haaland; worth a sanity check since Ipswich is newly-promoted and this pick leans on thin data.

**DEF:** Virgil van Dijk and Lacroix anchor the line — established starter and marquee signing respectively. N.Williams and Shaw add attacking upside from full-back. Bindon (Nott'm Forest) is a real name in the data but unverified by research — worth confirming he's actually first-team before matchday, since he wasn't part of the original DEF shortlist the research agent reviewed.

**MID:** Mbeumo, Enzo Fernández, Gakpo, and Szoboszlai all carry either penalty/set-piece duty or strong underlying numbers and confirmed starting roles. Enzo's inclusion still carries some risk flagged by research — he was missing from Chelsea's most recent predicted XI.

**FWD:** Haaland per your judgment call above. João Pedro is Chelsea's confirmed starting striker. Igor Jesus (Nottingham Forest) is in-form and a genuine starting contender per preseason reports.

**Notably excluded** despite reasonable model scores: Meslier, Heaton, Davies, Pecsi, Penders (all confirmed backups — Meslier's exclusion is now backed by his actual 2025/26 record of 0 starts, not just news), Chalobah (reportedly exiting for Como), Amass and Ramsay (buried in their depth charts), N.Jackson, Obi, and Welbeck (unlikely to feature in a reshaped Chelsea/Man Utd forward line).

## Fixtures watch (GW1)

Best fixtures on the board: **Arsenal vs Coventry** and **Man City vs Bournemouth** — both strong home favorites against weaker opposition, low-risk for clean sheets and attacking returns. **Hull vs Man Utd** favors United but Hull's own assets are a low-confidence unknown (no top-flight data). **Newcastle vs Liverpool** is the marquee toss-up of the round — avoid banking on a clean sheet from either side. Everything else on the card is close to a coin flip; preseason form and transfer hype are weak signals this early, so treat any "favorite" tag as a base rate, not a certainty.

## Chip strategy

**Hold everything.** Wildcard and Free Hit are mechanically blocked in GW1 anyway (you can't wildcard a squad you haven't picked). Bench Boost and Triple Captain are technically usable from GW1, but there's no blank/double gameweek or information edge this early to justify it. Save all four chips for when a real signal (fixture swing, DGW, squad crisis) shows up later in the season.

## Confidence notes

The model is now grounded in each player's actual 2025/26 Premier League record wherever one exists (80% of the pool), on top of a scoring-rules engine backtested against the full real 2025/26 season. The weakest links left are: the ~20% of the pool with no PL history at all (promoted-team players, brand new arrivals — the flat prior is the honest answer there, not a bug), and a couple of specific names (Bindon, the Ipswich "Palmer") that showed up through the numbers without a research agent having specifically vetted them yet, since the position agents were run against an earlier candidate list. Worth a final skim before the Friday deadline.
