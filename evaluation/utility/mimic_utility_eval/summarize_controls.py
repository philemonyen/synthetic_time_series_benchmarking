"""Turn controls/summary.json (from validate_pipeline.py) into a Markdown table."""
import json
import sys

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "controls/summary.json"))
f = lambda x, d=3: f"{x:.{d}f}"
L = ["| Control | Tier 1 non-inf / inconcl / inferior | Mean delta AUROC | W/IQR vitals (ref) | ACD vitals (ref) | "
     "Hourly-change W (ref) | Corr. diff (ref) | Glucose count | Discrim. AUROC (ref) | Recall at beta=1 (ref) | "
     "Authenticity (ref) | Tier 3 equivalent | MIA AUROC | Privacy loss | Below holdout p5 |",
     "|" + "---|" * 15]
for r in rows:
    L.append(f"| {r['control']} | {r['tier1_non_inferior']} / {r['tier1_inconclusive']} / {r['tier1_inferior']} of {r['tier1_n']} | "
             f"{f(r['mean_delta_auroc'])} | {f(r['w_per_iqr_vitals'])} ({f(r['w_per_iqr_vitals_ref'])}) | "
             f"{f(r['acd_vitals'])} ({f(r['acd_vitals_ref'])}) | {f(r['change_w_vitals'], 2)} ({f(r['change_w_vitals_ref'], 2)}) | "
             f"{f(r['corr_diff'])} ({f(r['corr_diff_ref'])}) | {r['glucose_median_count']:.0f} | "
             f"{f(r['discriminative_auroc'])} ({f(r['discriminative_auroc_ref'])}) | {f(r['recall_at_beta1'])} ({f(r['recall_at_beta1_ref'])}) | "
             f"{f(r['authenticity'])} ({f(r['authenticity_ref'])}) | {r['tier3_equivalent']} of {r['tier3_n']} | "
             f"{f(r['mia_auroc'])} | {f(r['privacy_loss'])} | {f(r['share_below_p5'])} |")
print("\n".join(L))
