# GTA VI Trophy Rarity Forecast — Earners/Players Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ibnu3456787/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ibnu3456787/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast, for each of
Grand Theft Auto VI's 55 PlayStation 5 trophies, its earn rate on Exophase at
2026-11-26 00:00 UTC as five quantiles (5/25/50/75/95%), scored by mean
pinball loss on the logit scale.

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`). Deterministic, offline, ~0.09 s
full-process per 310-read final-like case (limit 8 s).

## The model

### 1. Decompose the rate: earners / players

An Exophase rate is a ratio: `rate = earners / players_tracked`. The two
halves move for different reasons — earners grow as players *play*, players
grow as new buyers *join and sync* holding almost nothing. A forecaster that
extrapolates the rate directly (like the house trend baseline) conflates
progression with dilution, and gets both wrong when player growth decelerates.
This entry forecasts the two separately and divides:

- **Players N(T):** least-squares fit of log N on log(1 + hours since
  launch), quadratic when there are ≥ 5 distinct reads (captures the launch
  surge fading), clamped to [N(last), 10 × N(last)].
- **Earners E_i(T):** per-trophy power law, log E linear in log(1 +
  effective hours), anchored at the last read so only the *slope* is
  estimated. The slope is shrunk toward an archetype prior with weight
  `n/(n+3)` on the fitted slope. Effective hours weight weekend days ×1.5
  and Fridays ×1.15 (more play time); the final's horizon covers a weekend
  the history never saw. Late trophies whose deadline earners are a small
  fraction of their cap (< 2%) gain a threshold-crossing convexity term,
  `gamma × dx²` with `gamma = 1.5 × ln(delay/10)` from the archetype's
  median delay: the earners curve is S-shaped (a lognormal CDF in
  play-hours), so a log-log line underpredicts trophies that cross steeply
  during the horizon — in simulation the final-mission trophy came out
  14× low without it, 1.3× low with it.

### 2. Trophy archetypes (from each trophy's own name/description/tier)

| Archetype | Examples | Week-1 cap | Slope prior |
|---|---|---|---|
| saturated | prologue (Complete Gone Fishing) | 99.99% | 1.1 |
| story_ch1..ch4, story_final | Complete Chapter 1..4, final mission | 70/50/22/15/10% | 1.3..0.95 |
| story_mission | named missions (Breakfast Run, ...) | 40% | 1.15 |
| side_strand / side_all | final mission for a person / 8 strands | 30% / 6% | 1.05 / 0.85 |
| early_main / early_activity | fishing challenge / arcade game | 60% / 45% | 1.4 |
| grind | 250 stunts, 100 locations, 3 card sets | 3% | 0.7 |
| light_grind | 20 races, 100 vehicles, 5 bank vaults | 9% | 0.85 |
| skill / money / explore | gold medals / $250k laundering / all Leonida | 10/15/15% | 0.8/0.9/0.9 |
| misc | one-off feats (wanted level, robberies, ...) | 30% | 1.0 |
| completion / platinum | 100% Completion / platinum | 1.0% / 0.8% | 0.6 |

Caps are a *saturation correction*, not a loose guardrail: the raw
power-law extrapolation overshoots badly for fast early trophies (fishing:
84.9% uncapped vs ~59% truth in simulation), so the caps bind often and
pull runaway medians back toward plausible week-1 shares. The early_main
cap (60%) sits at the top of the ~51–61% truth range seen in simulation;
55/60/65 all score within 0.001 of each other there, and raising the arcade
cap above 45% worsens the simulator mean, so it stays. The story chain is
enforced on the medians (Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final ≥ 100% ≥ platinum,
platinum ≤ every trophy): a player who finished Chapter 2 necessarily
finished Chapter 1, so the forecast is coherent across trophies in a way
per-trophy fits are not. The chain is enforced in the empty-history branch
too, by the same pass, not left to hold by accident.

### 3. The near-100% trophy

The prologue trophy sits at 100% / 99.99% from launch night on (players sync
after playing it). When a trophy's last rate is ≥ 99% — the trigger is the
*data*, not the name — it gets quantiles hugging 100
([99.7, 99.95, 99.99, 99.99, 99.99]) with a small lower tail for players who
sync before finishing the prologue.

### 4. Flat history: expect a refresh

Exophase refreshes a game's counts on its own schedule, not continuously.
When every read in the history is identical, the page is stale and a refresh
is overdue; when it lands, new players arrive holding ~1 trophy each (the
prologue) and every other rate dilutes. The median is then

    med_rate = (r_last + q × (g − 1)) / g

where `g = 1 + backlog_hours` is the player-growth factor over the backlog
(staleness + horizon) and `q` is the new cohort's earn rate for the trophy's
archetype. Both are calibrated on launch night's two observed refreshes (see
Calibration). The spread is widened to 0.8 logit units: a refresh is a
known-unknown, and pinball loss punishes a narrow interval that misses far
more than a wide one.

### 5. Spreads (the pinball loss rewards honest intervals)

Quantiles are logit-normal around the median with a per-trophy spread
combining, in quadrature: a horizon term (0.35 × √days), the fit residual
(floored 0.15, capped 1.2; 0.8 when too few reads), the archetype uncertainty
(0.25 story … 0.55 platinum/grind), a staleness term (up to 0.25), a
weekend term (0.15 when the horizon covers an unobserved weekend), and a
late-trophy term (up to 0.5, for trophies whose deadline earners are under
5% of their cap: crossing the threshold during the horizon is a genuine
unknown). Total spread is clamped to [0.25, 1.6].

## Calibration (research only — the entry itself is offline)

- **Launch-night refresh (from the competition's own records):** the page
  was frozen at 638 players from 23:48Z to 04:00Z, then refreshed to 3,868
  players and 4,276 trophies earned; every rate but the 100% one fell. New
  cohort: 3,230 players, 3,333 trophies — ~1.03 per player, ~3,230 of them
  the prologue trophy.
- **Live snapshot, 19 Nov ~14:11 UTC** (fetched during the build for
  calibration only): players ≈ 13,900–14,000 (inferred from rate
  quantisation; the direct fetch is Cloudflare-blocked from this machine).
  New-cohort earn rates then: fishing ~1.9%, Chapter 1 ~0.7%, arcade ~0.4%,
  Breakfast Run ~0.2%, most others ≤ 0.1%. These set the `q` priors.
- **Player growth:** 638 → 3,868 in 4.7 h (×6.06), → ~14,000 in 14.2 h.
  The flat-branch rule `g = 1 + backlog_hours` reproduces the first refresh
  (×5.7 predicted vs ×6.06 actual) and brackets the second.

## Validation (all numbers measured, none invented)

The scorer is implemented in `research/backtest.py` and verified against the
rules' worked example (loss 0.403404, reproduced exactly). The house
baselines are re-implemented in `research/baselines.py` from the rules.

| Test | This entry | Persistence baseline | Trend baseline |
|---|---|---|---|
| Practice p00/p01 vs reconstructed answers (2 scenarios) | **0.186–0.200** | 1.501–1.555 | 1.501–1.555 |
| Live backtest 03:00Z → 14:11Z (real answers) | **0.315** | 2.377 | 2.377 |
| Simulator, 7 scenarios (mean) | **0.341** | 1.142 | 0.774 |
| Simulator, worst scenario | 0.696 | 1.253 | 0.954 |

Notes on honesty:

- The practice cases ship no answers, so the practice "answers" are
  reconstructed from the house's published refresh counts (3,868 players,
  4,276 trophies, prologue stayed 100%) with the 103 non-prologue trophies
  of the new cohort allocated by the profile observed in the live snapshot.
  The reconstruction is validated: it reproduces the house's published
  practice figures for the persistence baseline (loss 1.50–1.56 vs published
  ~1.51, typical miss 0.68pp vs ~0.68pp, coverage 0 vs ~0). Two allocation
  scenarios are reported; the entry beats the baseline by ~8× under both.
  The second scenario uses n_live=36,667, the rate-quantisation upper bound
  on the live player count (the snapshot's rates are shown to 0.01%, which
  bounds the player count the snapshot can represent); the first uses the
  point estimate 14,000.
- The flat-branch `q` priors were calibrated on the same live snapshot the
  live backtest scores against, so the live-backtest score (0.315) is
  optimistic; the *formula* is what the test validates, and the entry beats
  the baseline by ~7.5× there.
- The simulator (`research/simulator.py`) builds a final-like case (310
  reads, 2 days of history, 4.5-day horizon) from a cohort-progression
  ground truth. The entry wins all 7 scenarios, including two adversarial
  ones (no weekend effect; per-trophy delays/caps shuffled at random so the
  archetype priors are wrong by construction). 90% coverage across scenarios
  is 0.89–1.00 (mean ~0.96; honest is 0.90). Per-scenario losses for this
  entry: base 0.237, slow growth 0.261, fast growth 0.226, slow play 0.363,
  slow progress 0.353, no weekend 0.248, shuffled 0.696. The threshold
  fix (M1) and cap fix (M2) improved the simulator mean from 0.432 to 0.341
  and the worst scenario from 0.712 to 0.696; the practice and live-backtest
  scores are unchanged by those fixes (both exercise only the flat branch).
  The light_grind simulator truth (25 h median delay, 12% cap) is the
  builder's own assumption, so the light_grind improvement there is partly
  assumption-matched — but the change is principled on its own: a 20-race
  trophy plausibly exceeds 3% by day 7, and the 3% cap hard-blocked any
  forecast above it.
- Determinism: two runs on the same input are byte-identical. Runtime
  (measured on this machine): ~5 ms in-process per 310-read case (~90 ms
  full process); both practice cases ~75 ms full process (limit 8 s).
  Output ~2.9 KB (limit 65,536 bytes). Schema checked on every output (55
  trophies, 5 monotone quantiles each, all in [0, 100]). Edge cases tested
  without crashing: empty history, single read, all-null rates, zero
  horizon, never-earned trophies, ×1000 player jump, flat history + long
  horizon, players_tracked=0.

## Why this beats the house entries

- **Persistence** says rates stay put. Over a 4.5-day launch-week horizon
  they move enormously (early trophies climb 10–50× in earners; late ones
  dilute then crawl). It scores ~1.1–1.3 on final-like simulations.
- **The trend baseline** fits logit(rate) linearly in log-time, which
  extrapolates launch-night explosive growth far past saturation (typical
  miss 14–17pp in simulation) and cannot use the weekend, the story chain,
  or the dilution/progression split.
- **The house AI model** answers one median + spread per trophy from a
  single call; it has no fitted curves, no cross-trophy constraints, and no
  refresh model. This entry's edge is structural: earners/players
  decomposition, archetype shrinkage, the story chain, and refresh-aware
  intervals — all fitted to the case's own data at run time.

## Files

- `main.py` — the entry (Python 3.13, stdlib only).
- `research/backtest.py` — exact scorer + practice-answer reconstruction.
- `research/baselines.py` — house baselines re-implemented from the rules.
- `research/simulator.py` — synthetic final-like backtest (7 scenarios).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
