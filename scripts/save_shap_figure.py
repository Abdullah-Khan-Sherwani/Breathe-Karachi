"""Fetch latest SHAP from MongoDB and save bar chart to report_figures/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config.db import get_collection, COLLECTION_SHAP

OUT = Path(__file__).parent.parent / "report_figures"
OUT.mkdir(exist_ok=True)

doc = get_collection(COLLECTION_SHAP).find_one(
    {"model_type": "ensemble"}, sort=[("created_at", -1)]
)
if doc is None:
    raise RuntimeError("No SHAP document found in shap_explanations.")

entries   = doc["explanation"]           # list of {feature, importance}
features  = [e["feature"]  for e in entries]
importances = [e["importance"] for e in entries]
created   = doc.get("created_at", "")
print(f"SHAP doc created_at: {created}")
print("Top 5:", list(zip(features[:5], [f'{v:.4f}' for v in importances[:5]])))

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(features[::-1], importances[::-1], color="#4c72b0")
ax.set_xlabel("mean(|SHAP value|)  —  average impact on model output")
ax.set_title("SHAP Feature Importance (LightGBM ensemble, top 15)")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
plt.tight_layout()
fig.savefig(OUT / "fig_shap.png", bbox_inches="tight")
plt.close()
print(f"Saved  {OUT / 'fig_shap.png'}")
