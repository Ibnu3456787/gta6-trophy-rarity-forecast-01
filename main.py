"""gta6-trophy-rarity-forecast-01 entry: an earners/players forecaster.

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case's target as
five quantiles, scored by pinball loss on the logit scale.

Model (see README.md for the full reasoning):
  1. Decompose every rate into earners E(t) = rate * players / 100 and the
     players-tracked count N(t), and forecast the two separately. The rate is
     E(T) / N(T): this separates real player progression (earners grow as
     people play) from dilution (new players join holding almost nothing),
     which a direct extrapolation of the rate conflates.
  2. N(T): least-squares fit of log N on log(1 + hours since launch), quadratic
     when there are enough distinct reads (captures the launch surge fading).
  3. E_i(T): per-trophy power law, log E linear in log(1 + effective hours),
     anchored at the last read; the slope is shrunk toward an archetype prior
     (story chapters, side strands, grinds, money, misc, ... classified from
     each trophy's own name/description/tier). Effective hours weight weekend
     days heavier (more play time). Archetype ceilings cap runaway growth, and
     the story chain (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final >= 100% >= platinum)
     is enforced on the medians: a player who finished Chapter 2 necessarily
     finished Chapter 1. Late trophies whose earners are a small fraction of
     their cap gain a threshold-crossing convexity term (gamma * dx^2, gamma
     from the archetype's median delay) plus a wider spread, because the
     lognormal earners curve is S-shaped and a log-log line underpredicts
     trophies that cross steeply during the horizon.
  4. Flat history (no distinct read: the page has not refreshed): expect a
     refresh, which dilutes rates as new players arrive; the median is shifted
     down by an archetype-aware dilution factor and the spread is widened.
  5. Quantiles are logit-normal around the median with a per-trophy spread
     combining horizon, fit residual, archetype uncertainty, staleness and the
     unobserved weekend. The near-100% prologue trophy gets a saturated branch.

Standard library only, deterministic, offline. Run by the grader as
`python3 -P -s /bot/main.py` (Python 3.13); locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Archetypes: (week-1 ceiling % of tracked players, slope prior beta for
# log E vs log(1 + effective hours), archetype spread in logit units,
# new-cohort earn rate q in %: the share of brand-new players who hold the
# trophy within their first hours, used only when the history is flat and a
# refresh is expected). The ceilings are a SATURATION CORRECTION, not a
# loose guardrail: the raw power-law extrapolation overshoots badly for
# fast early trophies (fishing: 84.9% uncapped vs ~59% truth in simulation),
# so the caps bind often and pull runaway medians back toward plausible
# week-1 shares (early_main 60% sits at the top of the ~51-61% truth range
# seen in simulation; 55/60/65 all score within 0.001 of each other there).
# The q values are calibrated on the two refreshes observed on launch night
# (see README.md). The old `dilute` column is gone: dilution is now computed
# from the growth factor, not assumed (med_rate = (r_last + q*(g-1)) / g).
# ---------------------------------------------------------------------------
ARCHETYPES = {
    "saturated": (99.99, 1.1, 0.15, 99.0),
    "story_ch1": (70.0, 1.30, 0.25, 0.60),
    "story_ch2": (50.0, 1.20, 0.28, 0.06),
    "story_ch3": (22.0, 1.10, 0.30, 0.02),
    "story_ch4": (15.0, 1.00, 0.32, 0.02),
    "story_final": (10.0, 0.95, 0.35, 0.02),
    "story_mission": (40.0, 1.15, 0.30, 0.10),
    "side_strand": (30.0, 1.05, 0.35, 0.02),
    "side_all": (6.0, 0.85, 0.45, 0.01),
    "early_main": (60.0, 1.40, 0.30, 1.50),
    "early_activity": (45.0, 1.40, 0.30, 0.35),
    "grind": (3.0, 0.70, 0.50, 0.01),
    "light_grind": (9.0, 0.85, 0.45, 0.01),
    "skill": (10.0, 0.80, 0.45, 0.01),
    "money": (15.0, 0.90, 0.45, 0.02),
    "explore": (15.0, 0.90, 0.45, 0.02),
    "misc": (30.0, 1.00, 0.40, 0.05),
    "completion": (1.0, 0.60, 0.55, 0.0),
    "platinum": (0.8, 0.60, 0.55, 0.0),
}

# Threshold-crossing (late trophies): median play-hours to earn, by
# archetype, for archetypes whose earners curve is still on its steep,
# S-shaped (lognormal-CDF) part at the deadline. A log-log line cannot
# represent that shape: it underpredicts trophies whose earners are near
# zero at the deadline but cross steeply during the horizon (in simulation
# the final-mission trophy came out 14x low). For a trophy of one of these
# archetypes whose deadline earners are a small fraction of its cap
# (frac < LATE_FRAC), the extrapolation gains a convexity term
# gamma * dx^2 with gamma = GAMMA_SCALE * ln(delay / 10), ramped to zero
# as frac approaches LATE_FRAC, and the spread gains LATE_SPREAD (in
# quadrature, ramped to zero at frac = 0.05): crossing the threshold is a
# genuine unknown, and pinball loss punishes a narrow interval that misses
# far more than a wide one. Archetypes not listed here (short delays, or
# earners already past the steep part) get no convexity term.
DELAY_HOURS = {
    "story_ch3": 20.0, "story_ch4": 35.0, "story_final": 55.0,
    "side_all": 45.0, "grind": 90.0, "light_grind": 25.0,
    "skill": 40.0, "money": 30.0, "explore": 35.0,
    "completion": 120.0, "platinum": 130.0,
}
GAMMA_SCALE = 1.5
LATE_FRAC = 0.02
LATE_SPREAD = 0.5

# Story-chain order for the isotonic (monotone) constraint on medians:
# a player holding a later trophy necessarily holds every earlier one.
CHAIN = ["story_ch1", "story_ch2", "story_ch3", "story_ch4", "story_final",
         "completion", "platinum"]

# Play-time weights by weekday (UTC): weekend days carry more play hours.
DAY_WEIGHT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.15, 5: 1.5, 6: 1.5}

# Flat-history refresh model: while the page is stale, the tracked player
# count grows roughly linearly in backlog hours at launch-phase rates
# (observed: x6.06 over a 4.7h backlog, x15-58 over a 14.4h backlog).
# Spread 0.8: the grid search on the two backtests favours 0.6, but those
# medians are partly circular (the q priors were calibrated on the same
# snapshot the live backtest scores against), so 0.8 buys insurance for an
# unseen refresh at a small cost (+0.03 on the optimistic backtests).
FLAT_SPREAD = 0.8


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp(pct)
    return math.log(c / (100 - c))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float):
    """Round to 4 decimals, half up (the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def effective_hours(t: datetime) -> float:
    """Play-weighted hours from launch to t: weekend days count more.

    Integrates DAY_WEIGHT over whole days plus the partial day. Launch day
    (Thu 19 Nov 2026) is a weekday (weight 1.0).
    """
    if t <= LAUNCH:
        return 0.0
    total = 0.0
    day_start = LAUNCH
    while day_start < t:
        nxt = datetime.combine(day_start.date() + timedelta(days=1),
                               time(0, 0), tzinfo=timezone.utc)
        end = min(nxt, t)
        total += DAY_WEIGHT[day_start.weekday()] * (end - day_start).total_seconds() / 3600.0
        day_start = end
    return total


def classify(trophy: dict) -> str:
    """Archetype from the trophy's own tier/name/description."""
    tier = (trophy.get("tier") or "").lower()
    text = ((trophy.get("name") or "") + " " + (trophy.get("description") or "")).lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in text:
        return "completion"
    m = re.search(r"complete chapter (\d)", text)
    if m:
        return {"1": "story_ch1", "2": "story_ch2", "3": "story_ch3",
                "4": "story_ch4"}.get(m.group(1), "story_mission")
    if "complete the final mission for" in text or "mission strands for strangers" in text:
        return "side_all" if "strands" in text else "side_strand"
    if "complete the final mission" in text:
        return "story_final"
    if re.search(r"complete (breakfast run|the cartel heist|child's play|headliners|vcia takedown|operation savage glory)", text):
        return "story_mission"
    if "complete gone fishing" in text:
        return "saturated"
    if re.search(r"fishing challenge", text):
        return "early_main"
    if re.search(r"arcade game", text):
        return "early_activity"
    if "register" in text:
        # Registering a handful of vehicles is a light errand, not a grind.
        return "misc"
    m = re.search(r"win (\d+) .* races", text)
    if m:
        # A small race count is a light grind (a week of casual play can
        # clear 20 races); a large one is a real grind.
        return "light_grind" if int(m.group(1)) <= 20 else "grind"
    m = re.search(r"(\d+)\s*(stunts|locations|challenges|vehicles|bank vaults)", text)
    if m:
        if "discover" in text:
            # Passive discovery (vehicles found while playing) is light
            # even at a count of 100.
            return "light_grind"
        return "light_grind" if int(m.group(1)) <= 20 else "grind"
    if re.search(r"(collectible card sets|gold medal in all|drone race at every|meditate at every spot)", text):
        return "grind" if "card sets" in text or "meditate" in text else "skill"
    if re.search(r"(earn \$|spend \$|purchase all garage)", text):
        return "money"
    if "explore all of" in text:
        return "explore"
    return "misc"


def distinct_reads(history):
    """Reads whose (players, rates) differ from the previous kept read."""
    out = []
    prev = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != prev:
            out.append(h)
            prev = key
    return out


def lsq(xs, ys):
    """Least squares y = a + b x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    if n > 2:
        sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, n


def forecast_players(distinct, target: datetime, n_last: float) -> float:
    """Forecast players tracked at the target from the distinct reads."""
    if n_last <= 0:
        return 0.0
    if len(distinct) < 2:
        return n_last
    xs, ys = [], []
    for h in distinct:
        n = h.get("players_tracked")
        if not n or n <= 0:
            continue
        xs.append(math.log1p(hours_since_launch(parse_time(h["at"]))))
        ys.append(math.log(n))
    if len(xs) < 2:
        return n_last
    x_t = math.log1p(hours_since_launch(target))
    # Quadratic in log-time when enough points (captures the surge fading);
    # otherwise linear. Solved via normal equations on [1, x, x^2].
    if len(xs) >= 5:
        sx = [1.0, x_t, x_t * x_t]
        # Build normal equations for the quadratic fit.
        powers = [[1.0, x, x * x] for x in xs]
        ata = [[sum(p[i] * p[j] for p in powers) for j in range(3)] for i in range(3)]
        aty = [sum(p[i] * y for p, y in zip(powers, ys)) for i in range(3)]
        try:
            coef = _solve3(ata, aty)
            log_n = sum(c * s for c, s in zip(coef, sx))
            # Sanity: quadratic must not bend below the linear fit's value by
            # much, and must stay above the last observed log N.
            a, b, _, _ = lsq(xs, ys)
            log_n_lin = a + b * x_t
            log_n = min(log_n, log_n_lin + 0.35)
            log_n = max(log_n, math.log(n_last))
        except ArithmeticError:
            a, b, _, _ = lsq(xs, ys)
            log_n = a + b * x_t
    else:
        a, b, _, _ = lsq(xs, ys)
        log_n = a + b * x_t
    # Clamp in log space BEFORE exp: a steep fitted slope (e.g. a x1000
    # player jump between reads) can push log_n past exp's range. The
    # clamp is exactly the [N(last), 10 x N(last)] guard, applied early.
    log_n = min(max(log_n, math.log(n_last)), math.log(10.0 * n_last))
    return math.exp(log_n)


def _solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination (raises on singular)."""
    m = [row[:] + [bb] for row, bb in zip(a, b)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ArithmeticError("singular")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))) / m[r][r]
    return x


def _enforce_chain(medians, arch):
    """Story-chain isotonic constraint on median values (earners or rates).

    A player holding a later trophy necessarily holds every earlier one, so
    medians must be non-increasing along CHAIN; platinum sits below every
    trophy. `None` entries (saturated branch) are skipped. Applied in both
    the empty-history branch and the main branch so the chain holds by
    construction, not by accident.
    """
    chain_idx = {}
    for i, a in enumerate(arch):
        if a in CHAIN:
            chain_idx.setdefault(a, []).append(i)
    prev_cap = None
    for a in CHAIN:
        for i in chain_idx.get(a, []):
            if medians[i] is None:
                continue
            if prev_cap is not None:
                medians[i] = min(medians[i], prev_cap)
            prev_cap = medians[i] if prev_cap is None else min(prev_cap, medians[i])
    # Platinum is below every trophy.
    plat = [i for i, a in enumerate(arch) if a == "platinum"]
    if plat:
        others = [m for j, m in enumerate(medians) if m is not None and j not in plat]
        if others:
            cap = min(others)
            for i in plat:
                medians[i] = min(medians[i], cap)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    arch = [classify(t) for t in trophies]

    if not history:
        # No data at all: archetype priors, very wide spread.
        meds = []
        for t, a in zip(trophies, arch):
            cap = ARCHETYPES[a][0]
            meds.append(99.99 if a == "saturated" else min(cap * 0.2, 5.0))
        _enforce_chain(meds, arch)
        rows = [_row(t["id"], med, 1.5) for t, med in zip(trophies, meds)]
        return {"trophies": rows}

    last = history[-1]
    n_last = last.get("players_tracked") or 0
    rates_last = last.get("rates") or []
    distinct = distinct_reads(history)
    n_distinct = len(distinct)
    flat = n_distinct < 2

    # Staleness: hours from the last CHANGE to the checkpoint.
    last_change_at = parse_time(distinct[-1]["at"]) if distinct else at
    stale_h = max(0.0, (at - last_change_at).total_seconds() / 3600.0)

    # Weekend in the horizon that the history never saw (history covers
    # Thu-Fri only for the final; Sat/Sun are unobserved). Sample every 6h
    # and also check the target instant itself, so a short horizon ending
    # on a weekend day is not missed.
    horizon_has_weekend = (
        target.weekday() in (5, 6)
        or any(
            (at + timedelta(hours=h)).weekday() in (5, 6)
            for h in range(0, int(case["horizon_minutes"] // 60) + 1, 6)
        )
    )
    s_weekend = 0.15 if horizon_has_weekend else 0.0

    n_target = forecast_players(distinct, target, n_last) if not flat else n_last

    # Per-trophy earners series on distinct reads.
    x_last_eff = math.log1p(effective_hours(at))
    x_target_eff = math.log1p(effective_hours(target))

    medians = []   # median earners at target
    spreads = []   # logit spread per trophy
    for i, t in enumerate(trophies):
        a = arch[i]
        cap_pct, beta_prior, s_arch, q_new = ARCHETYPES[a]
        r_last = rates_last[i] if i < len(rates_last) else None
        e_last = (r_last * n_last / 100.0) if (r_last is not None and n_last) else 0.0

        if r_last is not None and r_last >= 99.0:
            # Near-100% trophy (the prologue): stays ~100%; only a few
            # players sync before finishing it. The trigger is the DATA,
            # not the name: a trophy whose last rate is below 99% goes
            # through the normal branch even if its archetype is saturated.
            medians.append(None)
            spreads.append(0.0)
            continue

        if flat:
            # No trend information: expect a refresh; new players hold
            # almost nothing, so rates dilute toward the new cohort's
            # profile: med_rate = (r_last + q*(g-1)) / g, where g is the
            # player-growth factor over the backlog and q the new cohort's
            # earn rate for this trophy's archetype.
            backlog_h = stale_h + case["horizon_minutes"] / 60.0
            growth = min(1.0 + backlog_h, 60.0)
            med_rate = ((r_last if r_last is not None else 0.0)
                        + q_new * (growth - 1.0)) / growth
            # Earners implied by the diluted rate at the (unchanged) N.
            medians.append(max(med_rate * n_target / 100.0, 0.0))
            spreads.append(FLAT_SPREAD)
            continue

        # Fit log earners on log(1 + effective hours), recent window first.
        pts = []
        for h in distinct:
            rates = h.get("rates") or []
            n = h.get("players_tracked") or 0
            if i >= len(rates) or rates[i] is None or not n:
                continue
            e = rates[i] * n / 100.0
            if e <= 0:
                continue
            pts.append((math.log1p(effective_hours(parse_time(h["at"]))),
                        math.log(e)))
        # Prefer the last 36h of reads for the slope when there are enough.
        cutoff = math.log1p(max(0.0, effective_hours(at) - 36.0 * 1.2))
        recent = [p for p in pts if p[0] >= cutoff]
        use = recent if len(recent) >= 4 else pts
        _, beta_fit, sd, npts = lsq([p[0] for p in use], [p[1] for p in use])
        # Shrink the slope toward the archetype prior; more data -> less shrink.
        w = min(npts, 20) / (min(npts, 20) + 3.0) if npts else 0.0
        beta = w * beta_fit + (1.0 - w) * beta_prior
        beta = min(max(beta, 0.0), 3.0)

        if e_last <= 0:
            # Never earned in the data: tiny median from the archetype cap.
            med_e = min(cap_pct * 0.02, 0.05) * n_target / 100.0
            medians.append(med_e)
            # A late archetype with zero earners at the deadline may still
            # cross its threshold during the horizon: widen the spread.
            s_late = LATE_SPREAD if arch[i] in DELAY_HOURS else 0.0
            spreads.append(math.sqrt(0.35 ** 2 * horizon_days + s_arch ** 2 + 0.5 ** 2 + s_weekend ** 2 + s_late ** 2))
            continue

        # Threshold-crossing convexity (see DELAY_HOURS): how far the
        # trophy's deadline earners sit below its archetype cap.
        frac = (e_last / (cap_pct * n_last / 100.0)
                if (cap_pct > 0 and n_last > 0) else 1.0)
        delay = DELAY_HOURS.get(a, 0.0)
        gamma = 0.0
        s_late = 0.0
        if delay > 10.0 and frac < LATE_FRAC:
            gamma = (GAMMA_SCALE * math.log(delay / 10.0)
                     * (1.0 - frac / LATE_FRAC))
        if delay > 10.0 and frac < 0.05:
            s_late = LATE_SPREAD * (1.0 - frac / 0.05)
        dx = x_target_eff - x_last_eff
        log_e_t = math.log(e_last) + beta * dx + gamma * dx * dx
        # Clamp the log before exp: the median is capped at the archetype
        # share right below anyway, so this changes nothing except that a
        # very long horizon can no longer overflow exp.
        if cap_pct > 0 and n_target > 0:
            log_e_t = min(log_e_t, math.log(cap_pct * n_target / 100.0))
        med_e = math.exp(log_e_t)
        # Ceiling: archetype cap share of the target player base; floor: earners
        # never decrease.
        med_e = min(med_e, cap_pct * n_target / 100.0)
        med_e = max(med_e, e_last)
        medians.append(med_e)

        s_fit = min(max(sd, 0.15), 1.2) if npts >= 3 else 0.8
        s_refresh = 0.25 * min(1.0, stale_h / 12.0)
        s = math.sqrt((0.35 ** 2) * horizon_days + s_fit ** 2 + s_arch ** 2
                      + s_refresh ** 2 + s_weekend ** 2 + s_late ** 2)
        spreads.append(min(max(s, 0.25), 1.6))

    # Story-chain isotonic constraint on median earners (chain order).
    _enforce_chain(medians, arch)

    rows = []
    for i, t in enumerate(trophies):
        a = arch[i]
        if medians[i] is None:
            # Saturated trophy: quantiles hugging 100, with a small lower tail
            # for players who sync before finishing the prologue.
            rows.append({"id": t["id"], "q": [99.7, 99.95, 99.99, 99.99, 99.99]})
            continue
        med_rate = 100.0 * medians[i] / n_target if n_target > 0 else 0.0
        rows.append(_row(t["id"], med_rate, spreads[i]))
    return {"trophies": rows}


def _row(tid: int, median_rate: float, spread: float) -> dict:
    lg = logit(median_rate)
    qs = [round4(inv_logit(lg + z * spread)) for z in Z]
    # Enforce non-decreasing after rounding.
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    qs = [min(max(q, 0), 100) for q in qs]
    return {"id": tid, "q": qs}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
