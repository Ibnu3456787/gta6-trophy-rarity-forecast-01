"""Backtest harness for gta6-trophy-rarity-forecast-01 (RESEARCH ONLY, not part of the entry).

Implements the exact scorer from the rules (pinball loss on the logit scale),
reconstructs approximate practice answers from the house's published refresh
counts, and scores any forecast file against them.
"""
import json, math, os, sys
from pathlib import Path

# Kit location: env override, else the sibling kit directory next to this repo.
KIT_DIR = Path(os.environ.get(
    "MAKEFUN_KIT_DIR",
    Path(__file__).resolve().parent.parent.parent / "gta6-trophy-rarity-forecast-01-kit" / "gta6-trophy-rarity-forecast-01-kit"))

Z = [-1.645, -0.674, 0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

def clamp(x):
    return min(99.99, max(0.01, x))

def logit(pct):
    c = clamp(pct)
    return math.log(c / (100 - c))

def pinball_item(y, q, tau):
    d = logit(y) - logit(q)
    return 2 * max(tau * d, (tau - 1) * d)

def score_forecast(forecast, answers):
    """forecast: {"trophies":[{"id","q":[5]}]} ; answers: {id: rate}. Returns (loss, typical_miss_pp, coverage90)."""
    items, miss, cov = [], [], 0
    by_id = {t["id"]: t["q"] for t in forecast["trophies"]}
    for tid, y in answers.items():
        q = by_id[tid]
        for tau, qq in zip(LEVELS, q):
            items.append(pinball_item(y, qq, tau))
        miss.append(abs(y - q[2]))
        if q[0] <= y <= q[4]:
            cov += 1
    return sum(items)/len(items), sum(miss)/len(miss), cov/len(answers)

def check_rules_example():
    # Rules worked example: rows [10,15,20,25,35], y=22 and y=8 -> loss 0.403404
    fc = {"trophies":[{"id":1,"q":[10,15,20,25,35]},{"id":2,"q":[10,15,20,25,35]}]}
    loss, miss, cov = score_forecast(fc, {1:22.0, 2:8.0})
    print(f"rules example: loss={loss:.6f} (expect 0.403404) miss={miss:.2f} (expect 7.00) cov={cov} (expect 0.5)")
    assert abs(loss - 0.403404) < 1e-6, "scorer mismatch vs rules example"
    print("scorer VERIFIED against rules worked example")

# ---- live snapshot 2026-11-19 ~14:11Z (browser.open fetch, research only) ----
LIVE_RATES = [0.03,99.99,0.73,0.14,0.06,0.06,0.06,0.24,0.06,0.06,0.06,0.06,0.06,
              0.03,0.03,0.04,0.04,0.03,0.04,1.89,0.41,0.04,0.04,0.04,0.03,0.03,
              0.04,0.03,0.04,0.03,0.03,0.03,0.09,0.05,0.24,0.06,0.14,0.14,0.06,
              0.04,0.04,0.04,0.04,0.08,0.04,0.09,0.04,0.05,0.04,0.11,0.06,0.04,
              0.05,0.04,0.04]

def reconstruct_practice_answers(p00_path, n_live=14000):
    """Reconstruct the 04:30Z refresh read (p00/p01 answers).

    Known from competition.yaml: after refresh, players 638 -> 3,868,
    trophies earned 943 -> 4,276; the 100% trophy stayed 100%.
    New cohort: 3,230 players, 3,333 trophies; 3,230 of them the prologue
    trophy, leaving 103 non-prologue trophies. Allocate those 103 across
    trophies in proportion to the new cohort's non-prologue distribution
    observed in the live 14:11Z snapshot (shares from that snapshot at
    assumed players n_live; N cancels in shares except for the launch
    earners subtracted, so we test sensitivity to n_live).
    """
    case = json.load(open(p00_path))
    last = case["history"][-1]["rates"]
    launch_e = [round(r * 638 / 100) for r in last]
    # new-cohort earners at 14:11Z
    live_e = [r * n_live / 100 for r in LIVE_RATES]
    new_e = [max(0.0, le - oe) for le, oe in zip(live_e, launch_e)]
    new_e[1] = 0.0  # prologue handled separately
    tot = sum(new_e)
    shares = [x / tot for x in new_e]
    alloc = [0] * 55
    # largest-remainder allocation of 103
    raw = [s * 103 for s in shares]
    base = [int(math.floor(x)) for x in raw]
    rem = 103 - sum(base)
    order = sorted(range(55), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rem]:
        base[i] += 1
    alloc = base
    answers = {}
    for i, t in enumerate(case["trophies"]):
        if i == 1:
            answers[t["id"]] = 100.0
        else:
            k = launch_e[i] + alloc[i]
            answers[t["id"]] = round(100 * k / 3868, 2)
    return answers, alloc, launch_e

if __name__ == "__main__":
    check_rules_example()
    kit = KIT_DIR
    for n_live in (14000, 36667):
        ans, alloc, le = reconstruct_practice_answers(kit / "input/p00.in", n_live)
        print(f"\n--- n_live={n_live}: reconstructed 04:30Z answers (top moves) ---")
        case = json.load(open(kit / "input/p00.in"))
        last = case["history"][-1]["rates"]
        for i, t in enumerate(case["trophies"]):
            if alloc[i] or last[i] > 1.0:
                print(f'  id {t["id"]:>2} {t["name"][:26]:<26} last={last[i]:>5} -> ans={ans[t["id"]]:>5} (old_e={le[i]}, +{alloc[i]})')
        print("  sum alloc:", sum(alloc), "| sum implied earned:",
              sum(round(a*3868/100) for a in ans.values()))
