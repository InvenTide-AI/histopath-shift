"""Write the GPU smoke-test plans (every arm, 60 steps) for the smoke caches.
  python src/v2/make_smoke_plan.py  -> $V2/smoke/plans/{C,S}_c17_fold0.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from make_plan import TIER_C_ARMS, TIER_S_ARMS, LADDER, TTA  # noqa: E402

STEPS = int(os.environ.get("SMOKE_STEPS", 60))
out = os.path.join(C.V2, "smoke", "plans")
os.makedirs(out, exist_ok=True)
for tier, arms in (("C", TIER_C_ARMS), ("S", TIER_S_ARMS + [a for a in LADDER if a not in TIER_S_ARMS])):
    jobs = []
    for arm in arms:
        hp = {"steps": STEPS}
        if arm in ("simclr", "simclr_sam", "compute_matched"):
            hp["ssl_epochs"] = 1
        jobs.append(dict(arm=arm, seed=0, hp=hp, tta=TTA[tier].get(arm, []), pcam=True, phase="smoke"))
    if tier == "C":   # TENT on ERM is tier-C only in the plan; also exercise bnadapt+tent on tier S ERM
        pass
    else:
        jobs[0]["tta"] = ["bnadapt", "tent"]
    p = os.path.join(out, f"{tier}_c17_fold0.json")
    json.dump(dict(tier=tier, cohort="c17", fold=0, phase="smoke", jobs=jobs), open(p, "w"), indent=1)
    print(p, len(jobs), "jobs")
