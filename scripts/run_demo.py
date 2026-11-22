"""End-to-end demo on synthetic data. Run: python scripts/run_demo.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retail_ds import business_case as bc
from retail_ds.data import make_daily_sales, make_transactions
from retail_ds.experiments import sample_size_per_arm
from retail_ds.forecasting import add_features, rolling_origin_eval
from retail_ds.segmentation import build_rfm, segment_customers, segment_profile

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

print("== I2: customer segmentation ==")
seg = segment_customers(build_rfm(make_transactions()))
prof = segment_profile(seg)
print(prof)
prof.to_csv(OUT / "segment_profile.csv")

print("\n== I1: forecasting (rolling origin) ==")
df = add_features(make_daily_sales())
res = rolling_origin_eval(df, ["2026-03-01", "2026-04-15", "2026-06-01"])
print(res)
res.to_csv(OUT / "forecast_eval.csv", index=False)

print("\n== A/B test sizing ==")
print("Visitors per arm:", sample_size_per_arm(0.032, 0.034))

print("\n== Business case ==")
base = bc.evaluate()
print(f"Run-rate benefit ${base['run_rate']:.2f}M | NPV ${base['npv']:.2f}M | "
      f"ROI {base['roi']:.0%} | payback {base['payback']} | break-even {bc.breakeven_factor():.0%}")
factors = [0.5, 0.75, 1.0, 1.25]
plt.figure(figsize=(6, 4))
plt.bar([f"{int(f*100)}%" for f in factors], [bc.evaluate(benefit_factor=f)["npv"] for f in factors], color="#2A7F7F")
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("Benefit realisation"); plt.ylabel("NPV ($M)"); plt.title("Sensitivity of NPV")
plt.tight_layout(); plt.savefig(OUT / "npv_sensitivity.png", dpi=150)
print("Outputs saved to outputs/")
