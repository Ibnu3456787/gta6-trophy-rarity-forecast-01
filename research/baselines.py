"""House baselines re-implemented from the rules/competition.yaml (research only)."""
import json, math
Z = [-1.645, -0.674, 0, 0.674, 1.645]

def clamp(x): return min(99.99, max(0.01, x))
def logit(p): c=clamp(p); return math.log(c/(100-c))
def inv_logit(l): return 100/(1+math.exp(-l))
def round4(x):
    s=x*10000; w=math.floor(s)
    if s-w>=0.5: w+=1
    v=w/10000
    return int(v) if float(v).is_integer() else v

def persistence(case):
    """House baseline = kit starter: median = last rate, spread 0.5*sqrt(days)."""
    last=case["history"][-1]
    spread=0.5*math.sqrt(case["horizon_minutes"]/1440)
    rows=[]
    for i,t in enumerate(case["trophies"]):
        r=last["rates"][i]
        lg=logit(r if r is not None else 0)
        rows.append({"id":t["id"],"q":[round4(inv_logit(lg+z*spread)) for z in Z]})
    return {"trophies":rows}

def trend(case):
    """House curve baseline: per-trophy LSQ line of logit(rate) on ln(1+hours
    since 2026-11-19T00:00Z) over last 24h of reads, read at target; spread =
    random walk + fit residual combined; <3 reads -> persistence."""
    import datetime as dt
    t0=dt.datetime(2026,11,19,0,0,tzinfo=dt.timezone.utc)
    at=dt.datetime.fromisoformat(case["at"].replace("Z","+00:00"))
    tgt=dt.datetime.fromisoformat(case["target"].replace("Z","+00:00"))
    win=[h for h in case["history"] if (at-dt.datetime.fromisoformat(h["at"].replace("Z","+00:00"))).total_seconds()<=24*3600]
    spread_rw=0.5*math.sqrt(case["horizon_minutes"]/1440)
    rows=[]
    for i,t in enumerate(case["trophies"]):
        pts=[]
        for h in win:
            r=h["rates"][i]
            if r is None: continue
            hrs=(dt.datetime.fromisoformat(h["at"].replace("Z","+00:00"))-t0).total_seconds()/3600
            pts.append((math.log1p(max(0.0,hrs)), logit(r)))
        if len(pts)<3:
            last=case["history"][-1]["rates"][i]
            lg=logit(last if last is not None else 0)
            rows.append({"id":t["id"],"q":[round4(inv_logit(lg+z*spread_rw)) for z in Z]})
            continue
        n=len(pts); mx=sum(p[0] for p in pts)/n; my=sum(p[1] for p in pts)/n
        sxx=sum((p[0]-mx)**2 for p in pts); sxy=sum((p[0]-mx)*(p[1]-my) for p in pts)
        b=sxy/sxx if sxx>0 else 0.0; a=my-b*mx
        resid=[p[1]-(a+b*p[0]) for p in pts]
        sd=math.sqrt(sum(r*r for r in resid)/max(1,n-2)) if n>2 else 0.0
        hrs_t=(tgt-t0).total_seconds()/3600
        mu=a+b*math.log1p(max(0.0,hrs_t))
        # "spread the random walk's and the fit's residual combined": quadrature
        spread=math.sqrt(spread_rw**2+sd**2)
        rows.append({"id":t["id"],"q":[round4(inv_logit(mu+z*spread)) for z in Z]})
    return {"trophies":rows}
