# FPL Squad Advisor -- Web App

A public-facing website for the squad advisor: expected points, transfer
recommendations, and player comparisons, built on top of the existing
model code (nothing was reimplemented -- the site just calls into
`fpl_client`, `xp_model`, `fixture_run`, `squad_optimizer`, and
`orchestrator`).

## Pages

- `/` -- home page, top xP picks, team ID lookup (or photo upload)
- `/finder` -- pick a position and a price ceiling, get the best
  projected picks that fit (optionally aware of a real squad via team ID,
  so it won't suggest a club you're already maxed out on)
- `/players` -- search for a player by name (accent-insensitive) to see
  their full profile, or add up to four same-position players to compare
  them head to head
- `/player/<id>` -- one player's page: percentile rank vs. their
  position, GW1-4 projection, last season, and price-range alternatives
- `/analytics` -- 2025/26 season analytics: value leaders, what actually
  predicts FPL points beyond the obvious stats, and more
- `/ai-performance` -- the AI manager's own season: a fully autonomous
  manager built on this project's own model, making real transfer and
  chip decisions gameweek by gameweek as the season progresses. See
  `agents/ai_manager.py` for how it decides.
- `/team/<id>` -- your own squad: pitch view with next-4-gameweek xP per
  player, weakest links with suggested alternatives, and a costed
  transfer recommendation

## Run it locally

This needs Python 3.9+. From the `webapp` folder:

```
pip install -r requirements.txt
python3 app.py
```

Then open `http://localhost:8888` in a browser.

That's it for the core site -- the model data is already in the
project's `data/raw` folder, so nothing else needs downloading or
configuring.

### One extra step for photo-upload team import

The "upload a photo of your team" option on the home page reads player
names off the image using OCR (the `pytesseract` Python package, already
in `requirements.txt`, plus a separate program called `tesseract` that
does the actual text recognition -- pip can't install that part since
it's not a Python package). On a Mac:

```
brew install tesseract
```

If `tesseract` isn't installed, that one feature shows a clear error
message and everything else on the site keeps working normally --
team-ID lookup doesn't need it at all.

## One thing to know about the team lookup

The `/team/<id>` page calls FPL's public API to fetch a real manager's
squad by team ID -- no password, ever, just the numeric ID from their
`fantasy.premierleague.com` URL. That lookup needs a normal internet
connection. Running it on your own machine or on any real hosting
provider, this works as-is. If a live lookup ever fails (network
hiccup, wrong ID, FPL API down), the page falls back to a labelled demo
squad so it still renders something reviewable rather than an error
page.

## Putting it online

Right now this only runs on your machine. To get a real link you can
share, it needs to run on a host that keeps a Python process alive and
reachable. A few options that all support Tornado (a Python web
framework) the same way they'd support any other Python app:

**Render** ([render.com](https://render.com)) -- probably the easiest
starting point. Push this project to a GitHub repo, connect the repo on
Render, set the start command to `python3 webapp/app.py`, and it builds
and deploys automatically. Render's free tier gives 750 instance-hours
a month, which comfortably covers one small site running continuously,
but the free tier spins the app down after 15 minutes of no traffic and
takes 30-60 seconds to wake back up on the next visit -- fine for
sharing with friends, not for a snappy always-on demo. $7/month removes
that cold-start delay if it matters later ([Render
docs](https://render.com/docs/faq)).

**Railway** ([railway.app](https://railway.app)) -- similar
git-push-to-deploy flow, usage-based free credit rather than a flat
free tier.

**Fly.io** ([fly.io](https://fly.io)) -- more control over the server,
slightly more setup, free allowance for small always-on apps.

I'd suggest starting with Render since it needs the least setup, and
switching later if you outgrow the free tier's cold starts. Happy to
walk through the actual deploy step by step whenever you're ready --
it's mostly clicking "New Web Service," pointing it at the repo, and
waiting a few minutes.

## Known limitations of this v1

- Every server start makes one best-effort attempt to refresh
  `data/raw/` from the live FPL API before building the player pool
  (`_refresh_raw_data()` in `app.py`) -- if that fails (no network),
  it falls back to whatever's already cached on disk, same as always.
  This matters for the AI Performance page specifically: it can only
  ever see a newly-finished real gameweek if the app has been started
  (or restarted) since that gameweek finished.
- The AI manager's chip-timing decisions (`agents/ai_manager.py`) are
  simple, explicitly-documented threshold heuristics, not competitive
  FPL strategy -- `squad_optimizer.py` and `chip_strategy.py`
  deliberately never decided *when* to use a chip, only whether a
  transfer is worth it and how close a chip is to its use-it-or-lose-it
  deadline.
