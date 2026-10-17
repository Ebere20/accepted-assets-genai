"""
Phase 1 EMPIRICAL pilot (observed data, Lagos, 10 briefs).
Observed per-brief pilot records (time, revisions, quality, errors and costs).
Costs computed with one uniform labour
rate (US$15/h, the rate implied by the manual baseline) + observed API cost.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

seg = ["Students", "Young Professionals", "Working Mothers", "Fitness Enthusiasts", "Entrepreneurs",
       "Senior Adults", "Rural Communities", "Premium Customers", "Environmentally Conscious", "Corporate Buyers"]
bid = [f"B{i:03d}" for i in range(1, 11)]
obs = {
 "Manual":      dict(gen=[28,31,34,30,35,35,39,33,32,37], edit=[12,14,15,13,15,16,18,14,14,16],
                     rev=[2,2,2,2,2,2,3,2,2,2], q=[4.2,4.2,4.3,4.2,4.3,4.1,4.0,4.3,4.2,4.3], api=[0]*10),
 "Prompt-only": dict(gen=[3,3,3,3,3,3,4,3,3,3], edit=[15,16,18,16,18,19,21,18,17,18],
                     rev=[3,3,3,3,3,3,4,3,3,4], q=[3.7,3.7,3.6,3.6,3.6,3.5,3.3,3.7,3.6,3.6],
                     api=[.03,.03,.04,.03,.04,.03,.04,.04,.04,.04]),
 "RAG":         dict(gen=[4,4,4,4,4,4,5,4,4,4], edit=[9,10,11,10,11,12,14,11,10,12],
                     rev=[2,2,2,2,2,2,3,2,2,2], q=[4.1,4.1,4.2,4.1,4.1,4.0,3.9,4.2,4.1,4.1],
                     api=[.05,.05,.06,.05,.06,.05,.06,.06,.06,.06]),
 "RAG+HITL":    dict(gen=[4,4,4,4,4,4,5,4,4,4], edit=[6,7,8,7,8,9,10,8,8,9],
                     rev=[1,2,2,2,2,2,2,2,2,2], q=[4.5,4.5,4.6,4.5,4.5,4.4,4.3,4.6,4.5,4.5],
                     api=[.05,.05,.06,.05,.06,.05,.06,.06,.06,.06]),
}
rows = []
for wf, d in obs.items():
    for i in range(10):
        tot = d["gen"][i] + d["edit"][i]
        rows.append(dict(brief_id=bid[i], segment=seg[i], workflow=wf, market="Lagos", language="English",
                         content_type="Social Post", gen_or_draft_min=d["gen"][i], editing_min=d["edit"][i],
                         total_min=tot, revisions=d["rev"][i], quality=d["q"][i], api_cost_usd=d["api"][i],
                         labour_cost_usd=round(tot * 15 / 60, 2), total_cost_usd=round(tot * 15 / 60 + d["api"][i], 3),
                         data_status="OBSERVED (empirical pilot)"))
yor = dict(gen=[5,5,5,5,5,5,6,5,5,5], edit=[10,10,11,10,11,12,14,11,11,12], rev=[2,2,2,2,2,3,3,2,2,2],
           fact=[4.1,4.1,4.0,4.1,4.0,3.9,3.8,4.1,4.0,4.0], flu=[3.6,3.7,3.6,3.7,3.6,3.4,3.3,3.7,3.6,3.5],
           tone=[3.8,3.8,3.7,3.8,3.7,3.6,3.5,3.9,3.8,3.8], cult=[3.5,3.6,3.5,3.5,3.5,3.3,3.2,3.6,3.5,3.4])
for i in range(10):
    tot = yor["gen"][i] + yor["edit"][i]
    rows.append(dict(brief_id=bid[i], segment=seg[i], workflow="RAG+HITL", market="Lagos", language="Yoruba",
                     content_type="Social Post", gen_or_draft_min=yor["gen"][i], editing_min=yor["edit"][i],
                     total_min=tot, revisions=yor["rev"][i], factual_retention=yor["fact"][i], fluency=yor["flu"][i],
                     brand_tone=yor["tone"][i], cultural_fit=yor["cult"][i], data_status="OBSERVED (empirical pilot)"))
P = pd.DataFrame(rows)
P.to_csv("data/pilot_observed.csv", index=False)

E = P[P.language == "English"]
W = ["Manual", "Prompt-only", "RAG", "RAG+HITL"]
wide = lambda c: E.pivot(index="brief_id", columns="workflow", values=c)[W]
T, Q, C = wide("total_min"), wide("quality"), wide("total_cost_usd")
out = {"means": E.groupby("workflow")[["gen_or_draft_min", "editing_min", "total_min", "revisions", "quality",
                                        "api_cost_usd", "total_cost_usd"]].mean().reindex(W).round(3).to_dict(orient="index")}
rng = np.random.default_rng(1)
comp = {}
for a in ["Prompt-only", "RAG", "RAG+HITL"]:
    red = 1 - T[a].mean() / T["Manual"].mean()
    idx = rng.integers(0, 10, (10000, 10))
    bs = 1 - T[a].values[idx].mean(1) / T["Manual"].values[idx].mean(1)
    w = stats.wilcoxon(T[a], T["Manual"])
    tq = stats.ttest_rel(Q[a], Q["Manual"])
    wq = stats.wilcoxon(Q[a], Q["Manual"])
    dq = Q[a] - Q["Manual"]
    comp[f"{a} vs Manual"] = dict(time_reduction=100 * red, ci=list(np.percentile(100 * bs, [2.5, 97.5])),
                                  wilcoxon_time_p=float(w.pvalue), q_diff=float(dq.mean()),
                                  q_ci=[float(dq.mean() - stats.t.ppf(.975, 9) * dq.std(ddof=1) / np.sqrt(10)),
                                        float(dq.mean() + stats.t.ppf(.975, 9) * dq.std(ddof=1) / np.sqrt(10))],
                                  t_q=float(tq.statistic), p_q=float(tq.pvalue), wilcoxon_q_p=float(wq.pvalue),
                                  cost_reduction=float(100 * (1 - C[a].mean() / C["Manual"].mean())))
for a, b in [("RAG", "Prompt-only"), ("RAG+HITL", "RAG")]:
    comp[f"{a} vs {b}"] = dict(time_reduction=float(100 * (1 - T[a].mean() / T[b].mean())),
                               wilcoxon_time_p=float(stats.wilcoxon(T[a], T[b]).pvalue),
                               q_diff=float((Q[a] - Q[b]).mean()), wilcoxon_q_p=float(stats.wilcoxon(Q[a], Q[b]).pvalue))
out["paired"] = comp
out["friedman_time"] = dict(zip(["chi2", "p"], map(float, stats.friedmanchisquare(*[T[w] for w in W]))))
out["friedman_quality"] = dict(zip(["chi2", "p"], map(float, stats.friedmanchisquare(*[Q[w] for w in W]))))
Y = P[P.language == "Yoruba"]
eng_h = E[E.workflow == "RAG+HITL"].set_index("brief_id")
yh = Y.set_index("brief_id")
out["yoruba"] = dict(total=float(Y.total_min.mean()), edit=float(Y.editing_min.mean()), gen=float(Y.gen_or_draft_min.mean()),
                     fact=float(Y.factual_retention.mean()), flu=float(Y.fluency.mean()), tone=float(Y.brand_tone.mean()),
                     cult=float(Y.cultural_fit.mean()),
                     edit_vs_english_p=float(stats.wilcoxon(yh.editing_min, eng_h.editing_min).pvalue))
# pilot error audit & rater agreement - observed
out["errors"] = {"Prompt-only": [8, 5, 3, 16], "RAG": [2, 1, 1, 4], "RAG+HITL": [1, 0, 1, 2]}
out["kappa"] = {"Brand": 0.78, "Factual": 0.82, "Readability": 0.74, "Personalisation": 0.71, "Overall": 0.76}
json.dump(out, open("results/pilot_summary.json", "w"), indent=2)
print(json.dumps(out, indent=1))
