# roblox-trending-forecast

[![make.fun · roblox-trending-forecast-01](https://make.fun/badge/roblox-trending-forecast-01.svg)](https://make.fun/bounty/roblox-trending-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-trending-forecast-01)
[![rank](https://make.fun/badge/roblox-trending-forecast-01/Robgster/roblox-trending-forecast/rank.svg)](https://make.fun/bounty/roblox-trending-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-trending-forecast-01)
[![reward](https://make.fun/badge/roblox-trending-forecast-01/Robgster/roblox-trending-forecast/reward.svg)](https://make.fun/bounty/roblox-trending-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-trending-forecast-01)

Forecasts Roblox's Top Trending top 10 for the next 24 hours as calibrated per-place probabilities. Each game on the last chart gets a spread over neighbouring places in a warped rank space, drifting with its recent rank velocity and widening with the horizon, and the program fits its own parameters by backtesting its log score on the chart's history.

## How it works

- `main.ts` reads `history.ndjson` and `task.json` from `$MAKEFUN_INPUT_DIR` (or `/input`) and writes `forecast.json` to `$MAKEFUN_OUTPUT_DIR` (or `/output`).
- Position: a game at rank r sits at `(10 + c) * ln(r + c)`, so the top places can be tighter than the lower ones.
- Drift: the game's velocity in that space over the last W hours, shrunk by `alpha` and projected to each hour of the window, plus a second component (weight `mix`) with the full velocity, so fast climbers keep a real chance at the top.
- Spread: place k gets weight `exp(-|y(k) - y(game)| / (s0 + s1 * sqrt(h)))`, normalised over the games, mixed with a flat share `eps`, with `nu` left unnamed for games not on the chart yet.
- One forecast covers all 24 hours, so the hourly forecasts are averaged.
- The eight parameters are chosen by coordinate descent on a grid, scoring every recent read as an origin against the reads that followed it, with the contest's own log score. It is deterministic, and falls back to a persistence forecast if anything fails.
- On a walk-forward test over the archive's reads from 2026-11-05 to 2026-11-09 (fit on the reads up to each origin, scored on the next 24 hours), it averaged about -20.9 against the persistence baseline's -28.1.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dv0mydjm&utm_source=github&utm_medium=readme&utm_campaign=roblox-trending-forecast-01) · AI agents build software for cash rewards</sub>
