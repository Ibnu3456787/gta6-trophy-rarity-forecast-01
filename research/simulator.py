"""Synthetic final-like backtest (RESEARCH ONLY, not part of the entry).

Generates a plausible 7-day GTA VI launch from a cohort-progression ground
truth, cuts a final-like case at the deadline (2 days of 10-minute reads),
and scores the entry against the house baselines on the simulated target
read. This tests the NON-FLAT branch of the model, which the practice cases
cannot exercise. The simulator's parameters are assumptions (documented
below), so this is a self-consistency / machinery check, not proof of the
final score -- but it catches bugs and gross miscalibration.

Ground truth:
  - Players: N(t) = 638 * (1 + t_hours)^beta (power law; beta ~ 1.04 matches
    the observed 638 -> 3,868 in 4.7h and ~14k-37k at 14.2h).
  - Each join cohort plays at a lognormal play-rate (median PLAY_MED h/day);
    weekend days multiply play by 1.6.
  - Each trophy has a lognormal hours-to-earn distribution (median M_I by
    archetype) and an eventual cap share C_I of players who ever earn it.
    E_i(t) = sum over cohorts of joiners * C_I * P(delay < hours played).
  - The page refreshes every ~4h; reads every 10 min show the last refresh.
"""
import json, math, os, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RESEARCH_DIR))
import main as M
from backtest import score_forecast
from baselines import persistence, trend

# Kit location: env override, else the sibling kit directory next to this repo.
KIT_DIR = Path(os.environ.get(
    "MAKEFUN_KIT_DIR",
    REPO_ROOT.parent / "gta6-trophy-rarity-forecast-01-kit" / "gta6-trophy-rarity-forecast-01-kit"))

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)

# (median hours of play to earn, eventual cap share) per archetype.
TRUTH = {
    "saturated": (0.5, 0.999),
    "story_ch1": (4.0, 0.85),
    "story_ch2": (10.0, 0.70),
    "story_ch3": (20.0, 0.55),
    "story_ch4": (35.0, 0.45),
    "story_final": (55.0, 0.35),
    "story_mission": (12.0, 0.60),
    "side_strand": (18.0, 0.40),
    "side_all": (45.0, 0.15),
    "early_main": (2.0, 0.75),
    "early_activity": (3.0, 0.60),
    "grind": (90.0, 0.08),
    "light_grind": (25.0, 0.12),
    "skill": (40.0, 0.12),
    "money": (30.0, 0.20),
    "explore": (35.0, 0.18),
    "misc": (8.0, 0.45),
    "completion": (120.0, 0.03),
    "platinum": (130.0, 0.02),
}
SIGMA_DELAY = 0.7      # lognormal spread of hours-to-earn
PLAY_MED = 2.0         # median play hours/day per player
PLAY_SIGMA = 0.8       # lognormal spread of play-rate
PLAY_BINS = [(-1.28, 0.1), (-0.52, 0.2), (0.0, 0.4), (0.52, 0.2), (1.28, 0.1)]  # (z, weight)
WEEKEND_MULT = 1.6


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def day_weight(t, weekend_mult=WEEKEND_MULT):
    return weekend_mult if t.weekday() in (5, 6) else 1.0


def play_hours_between(t0, t1, rate_per_day, weekend_mult=WEEKEND_MULT):
    """Play hours a player with `rate_per_day` weekday hours accumulates in [t0, t1)."""
    total = 0.0
    cur = t0
    while cur < t1:
        nxt = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        end = min(nxt, t1)
        total += rate_per_day * day_weight(cur, weekend_mult) * (end - cur).total_seconds() / 86400.0
        cur = end
    return total


def build_case(beta=1.04, play_med=PLAY_MED, delay_mult=1.0, seed_trophies=None,
               truth_override=None, weekend_mult=WEEKEND_MULT):
    """Simulate the launch; return (case, answers {id: rate})."""
    trophies = seed_trophies
    archs = [M.classify(t) for t in trophies]

    def n_true(t):
        h = max(0.0, (t - LAUNCH).total_seconds() / 3600.0)
        return 638.0 * (1.0 + h) ** beta

    # Join cohorts: 2-hour buckets from launch to target.
    cohorts = []  # (join_time, size)
    t = LAUNCH
    prev = n_true(t)
    while t < TARGET:
        t2 = t + timedelta(hours=2)
        n2 = n_true(min(t2, TARGET))
        cohorts.append((t + timedelta(hours=1), n2 - prev))
        prev = n2
        t = t2

    # Refresh times: every 4h from launch.
    refreshes = []
    t = LAUNCH
    while t <= TARGET + timedelta(hours=1):
        refreshes.append(t)
        t += timedelta(hours=4)

    def state_at(t):
        """True (N, rates) at time t."""
        n = n_true(t)
        rates = []
        for idx, a in enumerate(archs):
            tid = trophies[idx]["id"]
            if a == "saturated":
                # Observed reality: the prologue trophy sits at 100%/99.99%
                # from launch night on (players sync after playing it).
                rates.append(99.99)
                continue
            if truth_override and tid in truth_override:
                med_h, cap = truth_override[tid]
            else:
                med_h, cap = TRUTH[a]
            med_h *= delay_mult
            e = 0.0
            for jt, size in cohorts:
                if jt >= t:
                    break
                for z, w in PLAY_BINS:
                    rate = play_med * math.exp(PLAY_SIGMA * z)
                    h_played = play_hours_between(jt, t, rate, weekend_mult)
                    if h_played <= 0:
                        continue
                    p = phi((math.log(h_played) - math.log(med_h)) / SIGMA_DELAY)
                    e += size * w * cap * p
            rates.append(round(100.0 * e / n, 2) if n > 0 else 0.0)
        return n, rates

    # Page state at each refresh; reads every 10 min show the last refresh.
    states = {r: state_at(r) for r in refreshes}
    reads = []
    slot = LAUNCH + timedelta(minutes=10)
    # early recording: 30-min cadence until 19 Nov 11:30, then 10-min
    while slot < DEADLINE:
        last_r = max(r for r in refreshes if r <= slot)
        n, rates = states[last_r]
        earned = sum(round(rt * n / 100.0) for rt in rates)
        reads.append({"slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "at": slot.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                      "players_tracked": int(round(n)),
                      "trophies_earned": int(earned),
                      "rates": rates})
        step = 30 if slot < datetime(2026, 11, 19, 11, 30, tzinfo=timezone.utc) else 10
        slot += timedelta(minutes=step)
    # Answer: first refresh in [target, target+1h)
    ans_r = min(r for r in refreshes if TARGET <= r < TARGET + timedelta(hours=1))
    n_ans, rates_ans = states[ans_r]
    answers = {t["id"]: r for t, r in zip(trophies, rates_ans)}
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00-sim",
        "at": DEADLINE.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int((TARGET - DEADLINE).total_seconds() // 60),
        "target": TARGET.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": reads,
        "notes": "simulated",
    }
    return case, answers, (n_true(DEADLINE), n_ans)


if __name__ == "__main__":
    kit = KIT_DIR
    trophies = json.load(open(kit / "input" / "p00.in"))["trophies"]
    scenarios = [
        ("base beta=1.04", dict()),
        ("slow growth beta=0.8", dict(beta=0.8)),
        ("fast growth beta=1.3", dict(beta=1.3)),
        ("slow play (1.2h/d)", dict(play_med=1.2)),
        ("slow progress x1.5 delays", dict(delay_mult=1.5)),
        ("no weekend effect", dict(weekend_mult=1.0)),
    ]
    # Adversarial: per-trophy delays/caps shuffled at random (archetype
    # priors wrong by construction) -- fixed seed for reproducibility.
    import random
    rng = random.Random(20261119)
    override = {}
    for t in trophies:
        a = M.classify(t)
        if a == "saturated":
            continue
        override[t["id"]] = (10 ** rng.uniform(math.log10(2), math.log10(120)),
                             10 ** rng.uniform(math.log10(0.03), math.log10(0.8)))
    scenarios.append(("shuffled delays/caps (adversarial)", dict(truth_override=override)))
    for name, kw in scenarios:
        case, ans, (n_dl, n_tg) = build_case(seed_trophies=trophies, **kw)
        print(f"\n=== {name}: N(deadline)={n_dl:,.0f} N(target)={n_tg:,.0f} reads={len(case['history'])} ===")
        print(f"    true target rates: Ch1={ans[3]} Ch2={ans[4]} final={ans[7]} plat={ans[1]} fish={ans[20]} grind15={ans[15]}")
        for label, fn in (("persistence", persistence), ("trend", trend), ("MINE", lambda c: M.forecast_case(c, "sim"))):
            loss, miss, cov = score_forecast(fn(case), ans)
            print(f"    {label:12s} loss={loss:.6f} miss={miss:.2f}pp cov90={cov:.2f}")
