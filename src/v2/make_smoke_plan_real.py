"""Real-data GPU smoke plans (short budgets) on a real v2 fold cache: exercises per-tile
Macenko with the train reference, auto SSL pre-training, compute_matched's 10-checkpoint
schedule, fp32 eval, TTA, PCam, probe feature caching, on real tiles.
  python src/v2/make_smoke_plan_real.py -> $V2/smoke/plans/real_{C,S}_c17_fold0.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

out = os.path.join(C.V2, "smoke", "plans")
os.makedirs(out, exist_ok=True)
plans = {
    "C": [dict(arm="erm", hp={"steps": 100}, tta=["bnadapt", "tent"]),
          dict(arm="macenko", hp={"steps": 100}), dict(arm="tia_style", hp={"steps": 100}),
          dict(arm="simclr", hp={"steps": 100, "ssl_epochs": 1}),
          dict(arm="simclr_sam", hp={"steps": 100, "ssl_epochs": 1}),
          dict(arm="compute_matched", hp={"steps": 100, "ssl_epochs": 1}),
          dict(arm="irm", hp={"steps": 100, "irm_lambda": 100.0})],
    "S": [dict(arm="erm", hp={"steps": 60}, tta=["bnadapt", "tent"]),
          dict(arm="macenko", hp={"steps": 60}),
          dict(arm="simclr", hp={"steps": 60, "ssl_epochs": 1}),
          dict(arm="erm_lunit_dino", hp={"steps": 60}),
          dict(arm="probe_resnet50_imagenet", hp={"steps": 200}),
          dict(arm="probe_resnet50_imagenet", hp={"steps": 200}, seed=1)],
}
for tier, jobs in plans.items():
    js = [dict(seed=j.pop("seed", 0), pcam=True, phase="smoke", **j) for j in jobs]
    p = os.path.join(out, f"real_{tier}_c17_fold0.json")
    json.dump(dict(tier=tier, cohort="c17", fold=0, phase="smoke", jobs=js), open(p, "w"), indent=1)
    print(p, len(js), "jobs")
