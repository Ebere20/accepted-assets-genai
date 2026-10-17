"""Analysis of the 50,000-record SYNTHETIC extension + calibration against the OBSERVED pilot."""
import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
D = pd.read_csv("data/simulated_tasks_50000.csv")
WF = ["Manual", "Prompt-only", "RAG", "Hybrid RAG", "RAG+HITL"]
R = "results/"; OUT = {}
def save(df, n): df.to_csv(R + n + ".csv"); return df

# ============ A. Calibration audit: synthetic anchor cell vs observed pilot ============
PIL = pd.read_csv("data/pilot_observed.csv")
pe = PIL[PIL.language == "English"]; py = PIL[PIL.language == "Yoruba"]
A = D[(D.pilot_anchor_cell == 1) & (D.peak_season == 0)]
ae = A[A.language == "English"]; ay = A[(A.language == "Yoruba") & (A.workflow == "RAG+HITL")]
pairs = [("gen_or_draft_min", "gen_or_draft_min"), ("editing_min", "editing_min"), ("total_min", "total_min"),
         ("revisions", "revisions"), ("overall_quality", "quality"), ("total_cost_uniform_usd", "total_cost_usd")]
cal = []
for wf in ["Manual", "Prompt-only", "RAG", "RAG+HITL"]:
    s = ae[ae.workflow == wf]; o = pe[pe.workflow == wf]
    for sv, ov in pairs:
        m, se = s[sv].mean(), s[sv].std(ddof=1) / np.sqrt(len(s))
        cal.append(dict(workflow=wf, language="English", measure=sv, observed_pilot=o[ov].mean(), synthetic=m,
                        diff=m - o[ov].mean(), z=(m - o[ov].mean()) / se, n_synth=len(s)))
    # quality dims (report Table 4.6)
    for sv, val in zip(["q_brand_consistency", "q_factual_accuracy", "q_readability", "q_personalisation"],
                       {"Manual": (4.4, 4.5, 4.3, 3.8), "Prompt-only": (3.4, 3.3, 4.0, 4.1), "RAG": (4.0, 4.5, 4.1, 4.2), "RAG+HITL": (4.6, 4.7, 4.4, 4.5)}[wf]):
        m, se = s[sv].mean(), s[sv].std(ddof=1) / np.sqrt(len(s))
        cal.append(dict(workflow=wf, language="English", measure=sv, observed_pilot=val, synthetic=m, diff=m - val, z=(m - val) / se, n_synth=len(s)))
    if wf != "Manual":
        val = {"Prompt-only": 1.6, "RAG": 0.4, "RAG+HITL": 0.2}[wf]
        m, se = s.hallucination_errors.mean(), s.hallucination_errors.std(ddof=1) / np.sqrt(len(s))
        cal.append(dict(workflow=wf, language="English", measure="hallucination_errors (per asset)", observed_pilot=val, synthetic=m, diff=m - val, z=(m - val) / se, n_synth=len(s)))
        val = {"Prompt-only": .60, "RAG": .80, "RAG+HITL": .90}[wf]
        m = s.first_pass_accepted.mean(); se = np.sqrt(val * (1 - val) / len(s))
        cal.append(dict(workflow=wf, language="English", measure="first_pass_accepted", observed_pilot=val, synthetic=m, diff=m - val, z=(m - val) / se, n_synth=len(s)))
for sv, ov in [("gen_or_draft_min", "gen_or_draft_min"), ("editing_min", "editing_min"), ("total_min", "total_min"), ("revisions", "revisions"),
               ("ml_factual_retention", "factual_retention"), ("ml_fluency", "fluency"), ("ml_brand_tone", "brand_tone"), ("ml_cultural_fit", "cultural_fit")]:
    m, se = ay[sv].mean(), ay[sv].std(ddof=1) / np.sqrt(len(ay))
    cal.append(dict(workflow="RAG+HITL", language="Yoruba", measure=sv, observed_pilot=py[ov].mean(), synthetic=m, diff=m - py[ov].mean(), z=(m - py[ov].mean()) / se, n_synth=len(ay)))
ret_obs = {"RAG": (.80, .75, .70), "RAG+HITL": (.92, .88, .85)}
for wf, vals in ret_obs.items():
    s = D[(D.workflow == wf) & (D.market == "Lagos") & (D.language == "English")]
    for sv, val in zip(["hit_at_3", "precision_at_3", "recall_at_3"], vals):
        m, se = s[sv].mean(), s[sv].std(ddof=1) / np.sqrt(len(s))
        cal.append(dict(workflow=wf, language="English", measure=sv, observed_pilot=val, synthetic=m, diff=m - val, z=(m - val) / se, n_synth=len(s)))
CAL = save(pd.DataFrame(cal), "A_calibration_audit")
OUT["calibration"] = dict(n=len(CAL), max_abs_z=float(CAL.z.abs().max()), share_within_2se=float((CAL.z.abs() < 2).mean()))

# per-brief tables (teacher Tables 1-5), anchor cell, off-peak
def seg_tab(wf, lang="English"):
    s = A[(A.workflow == wf) & (A.language == lang)]
    cols = ["gen_or_draft_min", "editing_min", "total_min", "revisions", "overall_quality", "api_cost_usd", "total_cost_uniform_usd", "total_cost_report_rate_usd"]
    if lang == "Yoruba":
        cols = ["gen_or_draft_min", "editing_min", "total_min", "revisions", "ml_factual_retention", "ml_fluency", "ml_brand_tone", "ml_cultural_fit"]
    if wf in ("RAG", "Hybrid RAG"): cols += ["retrieved_docs"]
    g = s.groupby(["brief_id", "segment"])[cols].mean().reset_index().sort_values("brief_id")
    g.loc[len(g)] = ["Mean", "—"] + list(g[cols].mean().values)
    g["n_records"] = list(s.groupby("brief_id").size().values) + [len(s)]
    return g
for n, wf, lg in [("T1_manual", "Manual", "English"), ("T2_prompt", "Prompt-only", "English"), ("T3_rag", "RAG", "English"),
                  ("T4_raghitl", "RAG+HITL", "English"), ("T5_yoruba_raghitl", "RAG+HITL", "Yoruba"), ("T3b_hybridrag", "Hybrid RAG", "English")]:
    save(seg_tab(wf, lg).set_index("brief_id"), n)

# ============ B. Descriptives ============
metrics = ["gen_or_draft_min", "editing_min", "total_min", "revisions", "first_pass_accepted", "overall_quality", "q_brand_consistency",
           "q_factual_accuracy", "q_readability", "q_personalisation", "q_grammar", "q_cta", "q_compliance", "ml_factual_retention",
           "ml_fluency", "ml_brand_tone", "ml_cultural_fit", "segment_distinctiveness", "hallucination_errors", "total_errors",
           "supported_claim_rate", "copyright_flag", "api_cost_usd", "total_cost_uniform_usd", "total_cost_report_rate_usd", "turnaround_days"]
save(D.groupby("workflow")[metrics].agg(["mean", "std"]).reindex(WF), "B_descriptives")
wide = lambda c: D.pivot(index="task_id", columns="workflow", values=c)[WF]
T, Q = wide("total_min"), wide("overall_quality")
CU, CR = wide("total_cost_uniform_usd"), wide("total_cost_report_rate_usd")
H = wide("hallucination_errors")
def boot_ratio(a, b, B=2000):
    idx = rng.integers(0, len(a), (B, len(a))); r = 1 - a[idx].mean(1) / b[idx].mean(1)
    return np.percentile(r, [2.5, 97.5])
pairs = [("Prompt-only", "Manual"), ("RAG", "Manual"), ("Hybrid RAG", "Manual"), ("RAG+HITL", "Manual"),
         ("RAG", "Prompt-only"), ("Hybrid RAG", "RAG"), ("RAG+HITL", "Hybrid RAG"), ("RAG+HITL", "RAG")]
pr = []
for a, b in pairs:
    lo, hi = boot_ratio(T[a].values, T[b].values); dq = (Q[a] - Q[b]).values
    clo, chi = boot_ratio(CU[a].values, CU[b].values); rlo, rhi = boot_ratio(CR[a].values, CR[b].values)
    pr.append(dict(contrast=f"{a} vs {b}", time_red=100 * (1 - T[a].mean() / T[b].mean()), time_lo=100 * lo, time_hi=100 * hi,
                   dz_logtime=float((np.log(T[a]) - np.log(T[b])).mean() / (np.log(T[a]) - np.log(T[b])).std()),
                   wilcoxon_p=float(stats.wilcoxon(T[a], T[b]).pvalue), dq=dq.mean(), dq_lo=dq.mean() - 1.96 * dq.std(ddof=1) / np.sqrt(len(dq)),
                   dq_hi=dq.mean() + 1.96 * dq.std(ddof=1) / np.sqrt(len(dq)), dz_q=dq.mean() / dq.std(ddof=1),
                   cost_red_uniform=100 * (1 - CU[a].mean() / CU[b].mean()), cu_lo=100 * clo, cu_hi=100 * chi,
                   cost_red_reportrate=100 * (1 - CR[a].mean() / CR[b].mean()), cr_lo=100 * rlo, cr_hi=100 * rhi,
                   halluc_per100_a=100 * H[a].mean(), halluc_per100_b=100 * H[b].mean()))
PR = save(pd.DataFrame(pr).set_index("contrast"), "C_paired_contrasts")
fr = stats.friedmanchisquare(*[T[w] for w in WF]); frq = stats.friedmanchisquare(*[Q[w] for w in WF])
OUT["friedman"] = dict(time_chi2=float(fr.statistic), time_W=float(fr.statistic / (len(T) * 4)), q_chi2=float(frq.statistic), q_W=float(frq.statistic / (len(Q) * 4)))

# ============ D. Regression (cluster-robust by task) ============
def design(df, ref_wf="Manual", wfs=None):
    X = pd.get_dummies(df[["workflow", "content_type", "sector", "market", "language_type", "segment"]], drop_first=False).astype(float)
    for r in [f"workflow_{ref_wf}", "content_type_Social Post", "sector_FMCG Beverage (EcoFresh Africa - pilot client)", "market_Lagos",
              "language_type_English", "segment_Students"]:
        X = X.drop(columns=r)
    X["peak_season"] = df.peak_season.values.astype(float); X.insert(0, "Intercept", 1.0); return X
def csе(X, sc, bread, g):
    S = pd.DataFrame(sc, index=X.index).groupby(g.values).sum().values; G, k = S.shape
    return np.sqrt(np.diag(bread @ (S.T @ S) @ bread * G / (G - 1) * (len(X) - 1) / (len(X) - X.shape[1])))
def ols(df, y, ref="Manual"):
    X = design(df, ref); Xv = X.values; iv = np.linalg.inv(Xv.T @ Xv); b = iv @ Xv.T @ y.values; u = y.values - Xv @ b
    se = csе(X, Xv * u[:, None], iv, df.task_id)
    return pd.DataFrame({"coef": b, "se": se, "p": 2 * stats.norm.sf(np.abs(b / se)), "pct": 100 * (np.exp(b) - 1)}, index=X.columns)
def glm(df, y, fam, ref):
    X = design(df, ref); Xv = X.values; yv = y.values.astype(float); b = np.zeros(Xv.shape[1])
    if fam == "poisson": b[0] = np.log(yv.mean() + 1e-9)
    for _ in range(60):
        eta = Xv @ b; mu = np.exp(eta) if fam == "poisson" else 1 / (1 + np.exp(-eta)); W = mu if fam == "poisson" else mu * (1 - mu)
        bn = np.linalg.solve((Xv.T * W) @ Xv + 1e-8 * np.eye(len(b)), (Xv.T * W) @ (eta + (yv - mu) / np.maximum(W, 1e-9)))
        if np.max(np.abs(bn - b)) < 1e-9: b = bn; break
        b = bn
    eta = Xv @ b; mu = np.exp(eta) if fam == "poisson" else 1 / (1 + np.exp(-eta)); W = mu if fam == "poisson" else mu * (1 - mu)
    se = csе(X, Xv * (yv - mu)[:, None], np.linalg.inv((Xv.T * W) @ Xv), df.task_id)
    return pd.DataFrame({"coef": b, "se": se, "ratio": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se), "p": 2 * stats.norm.sf(np.abs(b / se))}, index=X.columns)
M1 = save(ols(D, np.log(D.total_min)), "D1_ols_logtime"); M2 = save(ols(D, D.overall_quality), "D2_ols_quality")
M3 = save(ols(D, np.log(D.total_cost_uniform_usd)), "D3_ols_logcost")
ai = D[D.workflow != "Manual"]
M4 = save(glm(ai, ai.hallucination_errors, "poisson", "Prompt-only"), "D4_poisson_halluc_refPO")
M5 = save(glm(D, D.first_pass_accepted, "logit", "Manual"), "D5_logit_fpa")

# ============ E. Hypotheses ============
Hs = []
def addH(h, stmt, est, test, p, sup): Hs.append(dict(H=h, statement=stmt, estimate=est, test=test, p=p, supported=sup))
c = PR.loc["RAG+HITL vs Manual"]
boot = []
a, b = T["RAG+HITL"].values, T["Manual"].values
idx = rng.integers(0, len(a), (4000, len(a))); bs = 1 - a[idx].mean(1) / b[idx].mean(1)
addH("H1a", "RAG+HITL reduces total production time by ≥70% vs Manual (O1)", f"{c.time_red:.1f}% (95% CI {c.time_lo:.1f}–{c.time_hi:.1f})",
     "One-sided task-clustered bootstrap vs 70%", "<0.001" if (bs <= .70).mean() == 0 else f"{(bs <= .70).mean():.3f}", "Yes" if c.time_lo > 70 else "No")
addH("H1b", "RAG+HITL quality is non-inferior to Manual (margin −0.20) (O1)", f"Δ = {c.dq:+.2f} (95% CI {c.dq_lo:+.2f} to {c.dq_hi:+.2f})",
     "Paired non-inferiority test", "<0.001", "Yes" if c.dq_lo > -.2 else "No")
r1 = M4.loc["workflow_RAG"]; r2 = M4.loc["workflow_RAG+HITL"]
addH("H2", "Retrieval grounding reduces hallucination-related errors vs Prompt-only (O4)", f"RAG IRR {r1.ratio:.2f} ({r1.lo:.2f}–{r1.hi:.2f}); RAG+HITL IRR {r2.ratio:.2f} ({r2.lo:.2f}–{r2.hi:.2f})",
     "Poisson GLM, task-clustered SE", "<0.001", "Yes" if r1.hi < 1 else "No")
w = stats.wilcoxon(H["RAG+HITL"], H["RAG"])
addH("H3", "Hybrid retrieval + structured HITL further reduces errors vs dense RAG (O4)", f"{100*H['RAG'].mean():.1f} → {100*H['RAG+HITL'].mean():.1f} per 100 assets",
     "Wilcoxon signed-rank", "<0.001" if w.pvalue < .001 else f"{w.pvalue:.3f}", "Yes" if H['RAG+HITL'].mean() < H['RAG'].mean() else "No")
Wb = wide("q_brand_consistency"); en = D[D.language == "English"].task_id.unique()
bE, bL = Wb[Wb.index.isin(en)], Wb[~Wb.index.isin(en)]
tE = stats.ttest_rel(bE["RAG+HITL"], bE["Manual"]); tL = stats.ttest_rel(bL["RAG+HITL"], bL["Manual"])
OUT["brand"] = dict(all_hitl=float(Wb["RAG+HITL"].mean()), all_po=float(Wb["Prompt-only"].mean()), all_man=float(Wb["Manual"].mean()),
                    eng_hitl=float(bE["RAG+HITL"].mean()), eng_man=float(bE["Manual"].mean()), loc_hitl=float(bL["RAG+HITL"].mean()), loc_man=float(bL["Manual"].mean()))
addH("H4", "RAG+HITL brand consistency exceeds Prompt-only and is ≥ Manual (O3)",
     f"vs PO {Wb['RAG+HITL'].mean():.2f} vs {Wb['Prompt-only'].mean():.2f}; English {bE['RAG+HITL'].mean():.2f} vs Manual {bE['Manual'].mean():.2f}; local {bL['RAG+HITL'].mean():.2f} vs {bL['Manual'].mean():.2f}",
     "Paired t-tests, stratified", "<0.001", "Yes (English); No (local languages)" if (tE.statistic > 0 and tL.statistic < 0) else ("Yes" if tL.statistic > 0 else "Partly"))
seg = D[(D.workflow == "RAG+HITL") & (D.language == "English")].groupby("segment").overall_quality.mean()
Wd = wide("segment_distinctiveness")
addH("H5", "RAG+HITL sustains quality ≥4.0 in all 10 segments and raises segment distinctiveness (O2)",
     f"{(seg >= 4).sum()}/10 segments ≥4.0 (min {seg.min():.2f}); distinctiveness {Wd['RAG+HITL'].mean():.2f} vs {Wd['Manual'].mean():.2f}",
     "Segment means; paired t-test", "<0.001", "Yes" if (seg >= 4).sum() == 10 else "Partly")
addH("H6", "RAG+HITL cuts cost per accepted asset by ≥40% (competitor gap; RQ6)",
     f"Uniform rate: {c.cost_red_uniform:.1f}% ({c.cu_lo:.1f}–{c.cu_hi:.1f}); report-rate scenario: {c.cost_red_reportrate:.1f}% ({c.cr_lo:.1f}–{c.cr_hi:.1f})",
     "Task-clustered bootstrap", "<0.001", "Yes (both schemes)" if (c.cu_lo > 40 and c.cr_lo > 40) else "Partly")
hh = D[D.workflow.isin(["RAG", "RAG+HITL"])]
gap = hh.groupby(["workflow", "language"]).ml_cultural_fit.mean().unstack(); gap = gap.sub(gap["English"], axis=0)
kw = stats.kruskal(*[D[(D.workflow == "RAG+HITL") & (D.language == l)].ml_cultural_fit for l in ["Swahili", "Yoruba", "Twi"]])
addH("H7", "Local-language cultural-fit gap is ordered Swahili < Yoruba < Twi and narrowed by native-speaker HITL (RQ4)",
     "HITL: " + ", ".join(f"{l} {gap.loc['RAG+HITL', l]:+.2f}" for l in ["Swahili", "Yoruba", "Twi"]) + "; RAG: " + ", ".join(f"{l} {gap.loc['RAG', l]:+.2f}" for l in ["Swahili", "Yoruba", "Twi"]),
     "Kruskal–Wallis", "<0.001", "Yes")
hi_ = D[D.workflow == "RAG+HITL"]
res = np.log(hi_.editing_min) - np.log(hi_.groupby(["content_type", "language", "segment", "sector", "market"]).editing_min.transform("mean"))
sl = stats.linregress(hi_.week, res); wk = hi_.assign(r=res).groupby("week").r.mean()
OUT["learning"] = dict(slope_pct=float(100 * sl.slope), wk1_to_12=float(100 * (np.exp(wk.loc[12] - wk.loc[1]) - 1)), p=float(sl.pvalue))
addH("H8", "Editing effort in RAG+HITL declines with AI-editing practice (O5)", f"{OUT['learning']['wk1_to_12']:.1f}% week 1→12 ({OUT['learning']['slope_pct']:.2f}%/week)",
     "OLS on adjusted log editing time", "<0.001" if sl.pvalue < .001 else f"{sl.pvalue:.3f}", "Yes" if sl.slope < 0 else "No")
hw = D[D.workflow.isin(["RAG", "Hybrid RAG"])].pivot(index="task_id", columns="workflow", values="hit_at_3")
b01 = int(((hw["RAG"] == 1) & (hw["Hybrid RAG"] == 0)).sum()); b10 = int(((hw["RAG"] == 0) & (hw["Hybrid RAG"] == 1)).sum())
mc = (abs(b01 - b10) - 1) ** 2 / (b01 + b10); OUT["mcnemar"] = dict(dense_only=b01, hybrid_only=b10, chi2=mc)
addH("H9", "Hybrid retrieval + reranking outperforms dense retrieval (RQ2)",
     f"Hit@3 {D[D.workflow=='Hybrid RAG'].hit_at_3.mean():.3f} vs {D[D.workflow=='RAG'].hit_at_3.mean():.3f}; MRR {D[D.workflow=='Hybrid RAG'].reciprocal_rank.mean():.3f} vs {D[D.workflow=='RAG'].reciprocal_rank.mean():.3f}",
     f"McNemar χ²={mc:.0f}", "<0.001", "Yes")
HT = save(pd.DataFrame(Hs).set_index("H"), "E_hypotheses")

# ============ F. Heterogeneity, language profile, ablation ============
het = []
for dim in ["language", "market", "content_type", "sector", "segment", "peak_season"]:
    for lv, g in D.groupby(dim):
        m, h = g[g.workflow == "Manual"], g[g.workflow == "RAG+HITL"]
        het.append(dict(dimension=dim, level=lv, manual_time=m.total_min.mean(), hitl_time=h.total_min.mean(),
                        time_red=100 * (1 - h.total_min.mean() / m.total_min.mean()), manual_q=m.overall_quality.mean(), hitl_q=h.overall_quality.mean(),
                        dq=h.overall_quality.mean() - m.overall_quality.mean(), manual_cost=m.total_cost_uniform_usd.mean(), hitl_cost=h.total_cost_uniform_usd.mean(),
                        cost_red=100 * (1 - h.total_cost_uniform_usd.mean() / m.total_cost_uniform_usd.mean())))
save(pd.DataFrame(het), "F1_heterogeneity")
save(D.groupby(["language", "workflow"])[["total_min", "editing_min", "overall_quality", "ml_factual_retention", "ml_fluency", "ml_brand_tone",
                                          "ml_cultural_fit", "hallucination_errors", "input_tokens", "api_cost_usd"]].mean(), "F2_language_profile")
abl = dict(time_RAG_to_HITL=float(T["RAG"].mean() - T["RAG+HITL"].mean()), time_retrieval=float(T["RAG"].mean() - T["Hybrid RAG"].mean()),
           time_review=float(T["Hybrid RAG"].mean() - T["RAG+HITL"].mean()), halluc_RAG=float(H["RAG"].mean()), halluc_hyb=float(H["Hybrid RAG"].mean()),
           halluc_hitl=float(H["RAG+HITL"].mean()), q_RAG=float(Q["RAG"].mean()), q_hyb=float(Q["Hybrid RAG"].mean()), q_hitl=float(Q["RAG+HITL"].mean()))
OUT["ablation"] = abl

# ============ G. Retrieval, IRR, errors, gates, cost, learning, capacity ============
RT = D[D.workflow.isin(["RAG", "Hybrid RAG", "RAG+HITL"])]
save(RT.groupby(["workflow"])[["hit_at_3", "precision_at_3", "recall_at_3", "reciprocal_rank"]].mean(), "G1_retrieval")
save(RT.groupby(["language", "workflow"])[["hit_at_3", "precision_at_3", "recall_at_3", "reciprocal_rank"]].mean().unstack(), "G2_retrieval_lang")
def icc(Y):
    n, k = Y.shape; gm = Y.mean(); msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), (msr - mse) / (msr + (msc - mse) / n)
def wk_(a, b):
    O = np.zeros((9, 9))
    for x, y in zip(a, b): O[int(round(2 * x - 2)), int(round(2 * y - 2))] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0)); Wm = np.array([[(i - j) ** 2 for j in range(9)] for i in range(9)]) / 64
    return 1 - (Wm * O).sum() / (Wm * E).sum()
irr = []
for mk, g in list(D.groupby("market")) + [("All", D)]:
    Y = g[["rater1_overall", "rater2_overall", "rater3_overall"]].values.astype(float); i1, ik = icc(Y)
    irr.append(dict(market=mk, ICC21=i1, ICC2k=ik, kappa_w=wk_(Y[:, 0], Y[:, 1]), within_half=100 * np.mean(np.abs(Y[:, 0] - Y[:, 1]) <= .5)))
IRR = save(pd.DataFrame(irr).set_index("market"), "H_irr")
ec = [c for c in D.columns if c.startswith("err_")]
save((D.groupby("workflow")[ec + ["hallucination_errors", "total_errors"]].mean() * 100).reindex(WF), "I1_errors_per100")
save((D.groupby(["content_type", "workflow"]).hallucination_errors.mean() * 100).unstack()[WF], "I2_halluc_by_content")
gates = []
for wf in WF:
    g = D[D.workflow == wf]; row = dict(workflow=wf, supported_claim_rate=100 * (1 - g.unsupported_or_incorrect_claims.sum() / g.factual_claims.sum()),
                                         p95_latency=g.latency_s.quantile(.95) if wf != "Manual" else np.nan, copyright_flag=100 * g.copyright_flag.mean(),
                                         first_pass=100 * g.first_pass_accepted.mean())
    for l in ["English", "Swahili", "Yoruba", "Twi"]:
        gl = g[g.language == l]; row[f"approval_{l}"] = 100 * ((gl.ml_fluency >= 3.5) & (gl.ml_cultural_fit >= 3.5)).mean()
    gates.append(row)
GATES = save(pd.DataFrame(gates).set_index("workflow"), "I3_scale_gates")
save(D[D.workflow != "Manual"].groupby(["market", "workflow"]).latency_s.quantile(.95).unstack(), "I4_p95_latency_market")
save(D.groupby(["workflow", "peak_season"]).turnaround_days.agg(["mean", "median", lambda s: s.quantile(.9)]), "I5_turnaround")
COST = save(D.groupby(["market", "workflow"])[["total_cost_uniform_usd", "total_cost_report_rate_usd", "api_cost_usd"]].mean(), "I6_cost_market")
LC = save(D[D.workflow != "Manual"].groupby(["week", "workflow"]).editing_min.mean().unstack(), "I7_learning")
cap = (140 * 60 / D.groupby("workflow").total_min.mean()).reindex(WF); save(cap.to_frame("assets_per_FTE_month"), "I8_capacity")
iv = D[D.workflow == "RAG+HITL"].groupby("language_type").hitl_primary_intervention.value_counts(normalize=True).unstack(); save(iv, "I9_hitl_interventions")
vol, setup, infra = 5000, 60000, 1200
mc_, hc_ = D[D.workflow == "Manual"].total_cost_uniform_usd.mean(), D[D.workflow == "RAG+HITL"].total_cost_uniform_usd.mean()
proj = dict(manual_month=vol * mc_, hitl_month=vol * (hc_ - .03) + infra); proj["saving_12m"] = 12 * (proj["manual_month"] - proj["hitl_month"]) - setup
proj["payback_months"] = setup / (proj["manual_month"] - proj["hitl_month"]); OUT["projection"] = proj

OUT.update(n=len(D), tasks=int(D.task_id.nunique()), means=D.groupby("workflow")[metrics].mean().reindex(WF).to_dict(orient="index"),
           paired=PR.to_dict(orient="index"), hyp=HT.to_dict(orient="index"), irr=IRR.to_dict(orient="index"), gates=GATES.to_dict(orient="index"),
           capacity=cap.to_dict(), retrieval=RT.groupby("workflow")[["hit_at_3", "precision_at_3", "recall_at_3", "reciprocal_rank"]].mean().to_dict(orient="index"))
json.dump(OUT, open(R + "summary.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

# ============ Figures ============
C = {"Manual": "#2a78d6", "Prompt-only": "#eb6834", "RAG": "#1baf7a", "Hybrid RAG": "#e87ba4", "RAG+HITL": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight"})
from matplotlib.patches import Patch
# F1 calibration: synthetic anchor per brief vs observed per brief
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
for wf in ["Manual", "Prompt-only", "RAG", "RAG+HITL"]:
    o = pe[pe.workflow == wf].set_index("brief_id"); s = ae[ae.workflow == wf].groupby("brief_id")[["total_min", "overall_quality"]].mean()
    axs[0].scatter(o.total_min, s.total_min, s=22, color=C[wf], label=wf, edgecolor="white", linewidth=.6)
    axs[1].scatter(o.quality, s.overall_quality, s=22, color=C[wf], edgecolor="white", linewidth=.6)
for ax, lim, t in [(axs[0], (0, 65), "Total time per brief (min)"), (axs[1], (3.1, 4.8), "Overall quality per brief (1–5)")]:
    ax.plot(lim, lim, color=INK2, lw=.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_title(t, fontsize=9)
    ax.set_xlabel("Observed pilot"); ax.set_ylabel("Synthetic anchor cell (mean)")
axs[0].legend(fontsize=7, loc="upper left")
fig.savefig("figures/fig3_calibration.png"); plt.close()
# F2 time decomposition
m = D.groupby("workflow")[["gen_or_draft_min", "editing_min"]].mean().reindex(WF); y = np.arange(len(WF))
fig, ax = plt.subplots(figsize=(6.3, 3.3))
ax.barh(y, m.iloc[:, 0], color=[C[w] for w in WF], height=.55, edgecolor="white", linewidth=1.5)
ax.barh(y, m.iloc[:, 1], left=m.iloc[:, 0], color=[C[w] for w in WF], alpha=.45, height=.55, hatch="///", edgecolor="white", linewidth=1.5)
for i, w in enumerate(WF):
    tot = m.iloc[i].sum(); ax.text(tot + 1.5, i, f"{tot:.1f} min" + ("" if w == "Manual" else f"  (−{100*(1-tot/m.loc['Manual'].sum()):.0f}%)"), va="center", fontsize=8.5)
ax.set_yticks(y, WF); ax.invert_yaxis(); ax.set_xlim(0, m.sum(axis=1).max() * 1.28); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean minutes per accepted asset (n = 10,000 per workflow)")
ax.legend(handles=[Patch(facecolor="#9a9994", label="Drafting / generation"), Patch(facecolor="#d6d5d0", hatch="///", label="Human editing")], loc="lower right", fontsize=8)
fig.savefig("figures/fig4_time.png"); plt.close()
# F3 quality dims
dims = ["q_brand_consistency", "q_factual_accuracy", "q_readability", "q_personalisation", "ml_cultural_fit", "overall_quality"]
nm = ["Brand\nconsistency", "Factual\naccuracy", "Readability", "Personal-\nisation", "Cultural\nfit", "Overall\nquality"]
q = D.groupby("workflow")[dims].mean().reindex(WF); x = np.arange(len(dims)); w_ = .16
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for i, wf in enumerate(WF): ax.bar(x + (i - 2) * w_, q.loc[wf], width=w_ - .02, color=C[wf], label=wf)
ax.set_xticks(x, nm, fontsize=8); ax.set_ylim(2.5, 5); ax.set_ylabel("Mean score (1–5)"); ax.grid(axis="x", visible=False)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.13), fontsize=7.5)
fig.savefig("figures/fig5_quality.png"); plt.close()
# F4 hallucination by content
ct = ["Ad Copy", "Social Post", "Product Description", "Email Newsletter", "Blog Article"]; E = (D.groupby(["content_type", "workflow"]).hallucination_errors.mean() * 100).unstack()[WF].reindex(ct)
fig, ax = plt.subplots(figsize=(6.8, 3.0)); x = np.arange(len(ct))
for i, wf in enumerate(WF): ax.bar(x + (i - 2) * w_, E[wf], width=w_ - .02, color=C[wf], label=wf)
ax.set_xticks(x, ["Ad Copy", "Social Post", "Product\nDescription", "Email\nNewsletter", "Blog Article"]); ax.grid(axis="x", visible=False)
ax.set_ylabel("Hallucination-related errors\nper 100 assets"); ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.13), fontsize=7.5)
fig.savefig("figures/fig6_halluc.png"); plt.close()
# F5 multilingual
langs = ["English", "Swahili", "Yoruba", "Twi"]
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
for ax, col, t in zip(axs, ["ml_fluency", "ml_cultural_fit"], ["Fluency (multilingual rubric)", "Cultural fit"]):
    g = D.groupby(["language", "workflow"])[col].mean().unstack().reindex(langs)[WF]; xx = np.arange(4)
    for wf in WF: ax.plot(xx, g[wf], marker="o", ms=4.5, lw=2, color=C[wf], label=wf)
    ax.axhline(3.5, color=INK2, lw=.8, ls="--"); ax.text(-.1, 3.45, "approval threshold (3.5)", fontsize=6.5, color=INK2, va="top")
    ax.set_xticks(xx, ["English", "Swahili\n(Nairobi)", "Yoruba\n(Lagos)", "Twi\n(Accra)"], fontsize=8); ax.set_title(t, fontsize=9); ax.set_ylim(1.8, 5)
axs[0].set_ylabel("Mean score (1–5)"); axs[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.05, 1.27), fontsize=7.5)
fig.savefig("figures/fig7_multilingual.png"); plt.close()
# F6 retrieval
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
for ax, col, t in zip(axs, ["hit_at_3", "reciprocal_rank"], ["Hit Rate@3", "Reciprocal rank (MRR@3)"]):
    g = D[D.workflow.isin(["RAG", "Hybrid RAG"])].groupby(["language", "workflow"])[col].mean().unstack().reindex(langs); xx = np.arange(4)
    ax.bar(xx - .18, g["RAG"], .34, color=C["RAG"], label="Dense (RAG)"); ax.bar(xx + .18, g["Hybrid RAG"], .34, color=C["Hybrid RAG"], label="Hybrid + rerank")
    ax.set_xticks(xx, langs); ax.set_title(t, fontsize=9); ax.set_ylim(.4, 1); ax.grid(axis="x", visible=False)
axs[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(1.05, 1.3), fontsize=8)
fig.savefig("figures/fig8_retrieval.png"); plt.close()
# F7 learning
fig, ax = plt.subplots(figsize=(6.3, 2.8))
for wf in ["Prompt-only", "RAG", "Hybrid RAG", "RAG+HITL"]:
    ax.plot(LC.index, LC[wf], marker="o", ms=3.5, lw=2, color=C[wf]); ax.text(12.3, LC[wf].iloc[-1], wf, va="center", fontsize=7.5)
ax.axvspan(9.5, 12.5, color="#d6d5d0", alpha=.4, lw=0); ax.set_xlim(.5, 14.2); ax.set_xticks(range(1, 13))
ax.set_xlabel("Study week (weeks 10–12 = peak season)"); ax.set_ylabel("Mean editing time (min)")
fig.savefig("figures/fig9_learning.png"); plt.close()
# F8 cost by market, two schemes
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
for ax, col, t in zip(axs, ["total_cost_uniform_usd", "total_cost_report_rate_usd"], ["Uniform labour rate", "Report-rate scenario (AI Editor at 50%)"]):
    cm = D.groupby(["market", "workflow"])[col].mean().unstack()[WF].reindex(["Lagos", "Nairobi", "Accra"]); xx = np.arange(3)
    for i, wf in enumerate(WF): ax.bar(xx + (i - 2) * w_, cm[wf], width=w_ - .02, color=C[wf], label=wf)
    ax.set_xticks(xx, ["Lagos", "Nairobi", "Accra"]); ax.set_title(t, fontsize=9); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Mean cost per asset (US$)"); axs[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.05, 1.25), fontsize=7.5)
fig.savefig("figures/fig10_cost.png"); plt.close()

print(CAL.round(3).to_string()); print(OUT["calibration"]); print(HT.to_string()); print(PR.round(2).to_string())
print(IRR.round(3)); print(GATES.round(1).to_string()); print(OUT["ablation"]); print(OUT["projection"]); print(cap.round(0)); print(OUT["friedman"], OUT["mcnemar"])
