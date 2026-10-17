"""
GenAI Creative Engine - 50,000-record SIMULATED (synthetic) extension of the observed pilot
===================================================================================
Every value produced here is SIMULATED. It is anchored to the OBSERVED Phase-1
Lagos pilot (per-brief values, Tables 4.1, 4.3-4.8, 4.10, 4.18, 4.19 and 3.9 of
the preliminary technical report) and to effect directions reported in the
supporting literature. It must be labelled synthetic in any publication.

Design (paired across workflows):
  5 workflows x 10,000 tasks = 50,000 records
  tasks = 5 sectors x 10 segments x 5 content types x 2 languages x replicates,
          replicates: Lagos 10 (HQ / pilot site, 50% of volume), Nairobi 5, Accra 5
Anchor cell  = pilot client (EcoFresh, FMCG) x Lagos x English x Social Post x off-peak
               -> reproduces the observed per-brief pilot values in expectation.
Yoruba anchor = same cell in Yoruba under RAG+HITL (report Table 4.18).
"""
import itertools, numpy as np, pandas as pd

SEED = 20260930
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ pilot anchors (OBSERVED)
SEGS = ["Students", "Young Professionals", "Working Mothers", "Fitness Enthusiasts", "Entrepreneurs",
        "Senior Adults", "Rural Communities", "Premium Customers", "Environmentally Conscious", "Corporate Buyers"]
BID = [f"B{i:03d}" for i in range(1, 11)]
P = {  # per-brief observed values, Lagos English social post (report Tables 4.1, 4.3, 4.4, 4.5)
 "Manual":   dict(gen=[28,31,34,30,35,35,39,33,32,37], edit=[12,14,15,13,15,16,18,14,14,16], rev=[2,2,2,2,2,2,3,2,2,2],
                  q=[4.2,4.2,4.3,4.2,4.3,4.1,4.0,4.3,4.2,4.3], api=[0]*10, docs=[0]*10),
 "Prompt-only": dict(gen=[3,3,3,3,3,3,4,3,3,3], edit=[15,16,18,16,18,19,21,18,17,18], rev=[3,3,3,3,3,3,4,3,3,4],
                  q=[3.7,3.7,3.6,3.6,3.6,3.5,3.3,3.7,3.6,3.6], api=[.03,.03,.04,.03,.04,.03,.04,.04,.04,.04], docs=[0]*10),
 "RAG":      dict(gen=[4,4,4,4,4,4,5,4,4,4], edit=[9,10,11,10,11,12,14,11,10,12], rev=[2,2,2,2,2,2,3,2,2,2],
                  q=[4.1,4.1,4.2,4.1,4.1,4.0,3.9,4.2,4.1,4.1], api=[.05,.05,.06,.05,.06,.05,.06,.06,.06,.06], docs=[4,4,5,4,5,5,5,5,5,5]),
 "RAG+HITL": dict(gen=[4,4,4,4,4,4,5,4,4,4], edit=[6,7,8,7,8,9,10,8,8,9], rev=[1,2,2,2,2,2,2,2,2,2],
                  q=[4.5,4.5,4.6,4.5,4.5,4.4,4.3,4.6,4.5,4.5], api=[.05,.05,.06,.05,.06,.05,.06,.06,.06,.06], docs=[5]*10),
}
# Yoruba RAG+HITL per brief (Table 4.18)
YOR = dict(gen=[5,5,5,5,5,5,6,5,5,5], edit=[10,10,11,10,11,12,14,11,11,12], rev=[2,2,2,2,2,3,3,2,2,2],
           fact=[4.1,4.1,4.0,4.1,4.0,3.9,3.8,4.1,4.0,4.0], flu=[3.6,3.7,3.6,3.7,3.6,3.4,3.3,3.7,3.6,3.5],
           tone=[3.8,3.8,3.7,3.8,3.7,3.6,3.5,3.9,3.8,3.8], cult=[3.5,3.6,3.5,3.5,3.5,3.3,3.2,3.6,3.5,3.4])
# Quality-rubric dimension means (Table 4.6): brand, factual, readability, personalisation
QD = {"Manual": (4.4, 4.5, 4.3, 3.8), "Prompt-only": (3.4, 3.3, 4.0, 4.1), "RAG": (4.0, 4.5, 4.1, 4.2), "RAG+HITL": (4.6, 4.7, 4.4, 4.5)}
# Multilingual rubric, English (Table 4.19 RAG+HITL: 4.5/4.6/4.4/4.5); other arms = ASSUMPTION
ML = {"Manual": (4.5, 4.5, 4.4, 4.4), "Prompt-only": (3.4, 4.1, 3.4, 3.8), "RAG": (4.4, 4.2, 4.0, 4.0), "RAG+HITL": (4.5, 4.6, 4.4, 4.5)}
# Errors per 10 briefs (Tables 4.7 + 4.8 unified): unsupported, missing_cit, incorrect, tone, generic, cta, stat, outdated
ERR = {"Prompt-only": (8, 5, 3, 4, 3, 2, 3, 1), "RAG": (2, 1, 1, 1, 1, 0, 0, 0), "RAG+HITL": (1, 0, 1, 1, 0, 0, 0, 0),
       "Manual": (1.0, 1.0, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5)}           # Manual not audited in pilot -> ASSUMPTION
ETYPES = ["unsupported_claims", "missing_citations", "incorrect_facts", "wrong_tone", "generic_copy",
          "missing_cta", "unsupported_statistic", "outdated_claim"]
# Editing interventions per 10 briefs (Table 4.10): major, minor; first-draft acceptance
EDITS = {"Prompt-only": (3, 14, .60), "RAG": (1, 11, .80), "RAG+HITL": (1, 8, .90), "Manual": (1, 9, .70)}  # Manual = ASSUMPTION

# ------------------------------------------------------------------ ablation arm (ASSUMPTION)
# Hybrid RAG without structured HITL: hybrid retrieval assumed to deliver 40% of the RAG->RAG+HITL
# time/quality gain and 60% of the factual-error gain; review delivers the remainder.
def interp(a, b, w): return a + w * (b - a)
P["Hybrid RAG"] = dict(gen=P["RAG+HITL"]["gen"], edit=[interp(r, h, .4) for r, h in zip(P["RAG"]["edit"], P["RAG+HITL"]["edit"])],
                       rev=[interp(r, h, .4) for r, h in zip(P["RAG"]["rev"], P["RAG+HITL"]["rev"])],
                       q=[interp(r, h, .4) for r, h in zip(P["RAG"]["q"], P["RAG+HITL"]["q"])],
                       api=P["RAG+HITL"]["api"], docs=[5] * 10)
QD["Hybrid RAG"] = tuple(interp(a, b, .4) for a, b in zip(QD["RAG"], QD["RAG+HITL"]))
ML["Hybrid RAG"] = tuple(interp(a, b, .3) for a, b in zip(ML["RAG"], ML["RAG+HITL"]))
ERR["Hybrid RAG"] = tuple(interp(a, b, .6 if i < 3 else .2) for i, (a, b) in enumerate(zip(ERR["RAG"], ERR["RAG+HITL"])))
EDITS["Hybrid RAG"] = (1, 10, .84)
WF = ["Manual", "Prompt-only", "RAG", "Hybrid RAG", "RAG+HITL"]
STRUCTURED = {"Manual": False, "Prompt-only": False, "RAG": False, "Hybrid RAG": False, "RAG+HITL": True}

# ------------------------------------------------------------------ design factors
CONTENT = {  # manual draft/edit mult, AI gen/edit mult, api mult, error mult, AI quality offset, words, claims
 "Social Post":         dict(md=1.00, me=1.00, ag=1.00, ae=1.00, api=1.0, err=1.0, aq=0.00, words=175, claims=4),
 "Ad Copy":             dict(md=0.60, me=0.75, ag=0.85, ae=0.75, api=0.7, err=0.7, aq=0.02, words=50, claims=2),
 "Product Description": dict(md=0.90, me=1.00, ag=1.00, ae=1.05, api=1.0, err=1.2, aq=-0.02, words=120, claims=6),
 "Email Newsletter":    dict(md=1.60, me=1.50, ag=1.25, ae=1.70, api=1.7, err=1.5, aq=-0.05, words=300, claims=8),
 "Blog Article":        dict(md=3.20, me=2.60, ag=1.90, ae=3.60, api=3.8, err=2.3, aq=-0.10, words=800, claims=18),
}
SECTORS = {  # code, complexity, claims-risk (ARCON substantiation risk)
 "FMCG Beverage (EcoFresh Africa - pilot client)": ("C1", 1.00, 1.00),
 "Mobile Fintech": ("C2", 1.08, 1.40), "Telecom Data Services": ("C3", 0.96, 1.15),
 "Healthcare & Pharmacy Retail": ("C4", 1.12, 1.80), "Retail & E-commerce": ("C5", 0.92, 0.90),
}
MARKETS = {  # labour rate US$/h (Lagos from pilot; others ASSUMPTION), manual time, latency, disruption prob, reps
 "Lagos":   dict(country="Nigeria", local="Yoruba",  rate=15.00, mt=1.00, lat=1.15, disrupt=.08, reps=10),
 "Nairobi": dict(country="Kenya",   local="Swahili", rate=16.50, mt=0.98, lat=1.00, disrupt=.03, reps=5),
 "Accra":   dict(country="Ghana",   local="Twi",     rate=13.50, mt=1.02, lat=1.08, disrupt=.05, reps=5),
}
# Market x segment fit (localisation study): + means messaging resonates more strongly
FIT = {("Lagos", "Students"): .06, ("Lagos", "Young Professionals"): .06, ("Lagos", "Working Mothers"): .04,
       ("Nairobi", "Environmentally Conscious"): .12, ("Nairobi", "Fitness Enthusiasts"): .08, ("Nairobi", "Premium Customers"): .03,
       ("Accra", "Senior Adults"): .06, ("Accra", "Working Mothers"): .05, ("Accra", "Environmentally Conscious"): .04}
# Language (relative to English) - Yoruba OBSERVED (RAG+HITL), Swahili/Twi literature-extrapolated
Y = dict(gen=np.mean(YOR["gen"]) / 4.1, edit=np.mean(YOR["edit"]) / 8.0,
         d=(np.mean(YOR["fact"]) - 4.5, np.mean(YOR["flu"]) - 4.6, np.mean(YOR["tone"]) - 4.4, np.mean(YOR["cult"]) - 4.5))
LANG = {
 "English": dict(sev=0.0, gen=1.0, edit=1.0, tok=1.0, man=1.00, err=1.00, ret=0.00),
 "Swahili": dict(sev=0.55, gen=1 + .55 * (Y["gen"] - 1), edit=1 + .55 * (Y["edit"] - 1), tok=1.5, man=1.15, err=1.15, ret=.03),
 "Yoruba":  dict(sev=1.00, gen=Y["gen"], edit=Y["edit"], tok=1.9, man=1.25, err=1.30, ret=.06),
 "Twi":     dict(sev=1.30, gen=1 + 1.3 * (Y["gen"] - 1), edit=1 + 1.3 * (Y["edit"] - 1), tok=2.1, man=1.30, err=1.50, ret=.10),
}
WEEKS, PEAK = 12, {10, 11, 12}
EDITORS_PER_MARKET, RATERS = 8, 3
PRICE_IN, PRICE_OUT = 2.50e-6, 10.00e-6

# ------------------------------------------------------------------ latent effects
editors = {}
for m in MARKETS:
    for e in range(EDITORS_PER_MARKET):
        editors[f"{m[:3].upper()}-E{e+1:02d}"] = dict(market=m, speed=float(np.exp(rng.normal(-.005, .10))),
                                                      skill=float(rng.normal(0, .07)), exp=int(rng.integers(0, 13)))
by_mkt = {m: [k for k, v in editors.items() if v["market"] == m] for m in MARKETS}
for m, ids in by_mkt.items():   # centre editor random effects within market (log-speed mean 0, skill mean 0)
    ls = np.log([editors[i]["speed"] for i in ids]); sk = np.array([editors[i]["skill"] for i in ids])
    for i, a, b in zip(ids, ls - ls.mean(), sk - sk.mean()):
        editors[i]["speed"], editors[i]["skill"] = float(np.exp(a)), float(b)
rater_bias = {m: rng.normal(0, .06, RATERS) for m in MARKETS}
_raw = lambda w: 1.0 + .25 * np.exp(-(w - 1) / 3.0)
LEARN_NORM = np.mean([_raw(w) for w in range(1, 10)])
learn = lambda w, x: _raw(w) / LEARN_NORM * (1 - .006 * (x - 6))
ln = lambda sd: float(np.exp(rng.normal(-sd * sd / 2, sd)))
clip = lambda x: float(np.clip(x, 1, 5))

rows = []; tid = 0
combos = []
for mkt, M in MARKETS.items():
    for rep in range(1, M["reps"] + 1):
        for (sec, (cc, cx, risk)), si, ct, is_loc in itertools.product(SECTORS.items(), range(10), CONTENT, [0, 1]):
            combos.append((mkt, rep, sec, cc, cx, risk, si, ct, is_loc))
assert len(combos) == 10000

for (mkt, rep, sec, cc, cx, risk, si, ct, is_loc) in combos:
    tid += 1
    M, C = MARKETS[mkt], CONTENT[ct]
    seg, bid = SEGS[si], BID[si]
    lang = M["local"] if is_loc else "English"; L = LANG[lang]
    week = int(rng.integers(1, WEEKS + 1)); peak = week in PEAK
    anchor = (cc == "C1" and mkt == "Lagos" and ct == "Social Post")
    b_cx = ln(.10); tq = float(rng.normal(0, .08)); n_rel_kb = int(3 + (rng.random() < .35))
    fit = FIT.get((mkt, seg), 0.0) - (.05 if seg == "Rural Communities" and mkt != "Lagos" else 0)
    eds = rng.choice(by_mkt[mkt], 5, replace=False); order = rng.permutation(5) + 1

    for wi, wf in enumerate(WF):
        ed = editors[eds[wi]]; ai = wf != "Manual"; unstruct = ai and not STRUCTURED[wf]
        pb = P[wf]
        # ---------------- time ----------------
        if not ai:
            pk = 1.12 if peak else 1.0
            mult = C["md"] * cx * M["mt"] * L["man"] * pk * b_cx * ed["speed"]
            gen = pb["gen"][si] * mult * ln(.12)
            edit = pb["edit"][si] * (C["me"] / C["md"]) * mult * ln(.15)
        else:
            pk = 1.03 if peak else 1.0
            if wf == "RAG+HITL" and lang == "Yoruba":
                g0, e0 = YOR["gen"][si], YOR["edit"][si]          # observed Yoruba anchors
            else:
                g0, e0 = pb["gen"][si] * L["gen"], pb["edit"][si] * L["edit"]
            # pilot generation times already include Lagos's average disruption delay (8% x 5 min): remove it from the base
            gen = max(g0 - MARKETS["Lagos"]["disrupt"] * 5.0, 0.5) * C["ag"] * pk * ln(.12)
            if rng.random() < M["disrupt"]:
                gen += float(rng.uniform(2, 8))                    # power / connectivity disruption
            edit = e0 * C["ae"] * cx * pk * b_cx * ed["speed"] * learn(week, ed["exp"]) * ln(.15)
        gen, edit = round(gen, 1), round(edit, 1); total = round(gen + edit, 1)

        # ---------------- quality rubric ----------------
        sev = L["sev"] * (1.25 if unstruct else 1.0)
        if not ai:
            d_fact, d_flu, d_tone, d_cult = (0, -.10 * L["sev"], -.05 * L["sev"], .10 * min(L["sev"], 1)) if is_loc else (0, 0, 0, 0)
        else:
            d_fact, d_flu, d_tone, d_cult = [sev * x for x in Y["d"]]
        pkq = (-.15 if peak else 0) if not ai else (-.03 if peak else 0)
        caq = C["aq"] if ai else 0
        shock = tq + ed["skill"] + pkq + float(rng.normal(0, .06))
        seg_off = pb["q"][si] - np.mean(pb["q"])
        qb, qf, qr, qp = QD[wf]
        brand = clip(qb + seg_off + d_tone + caq + shock + rng.normal(0, .10))
        factual = clip(qf + seg_off + d_fact + caq + shock + rng.normal(0, .10))
        readab = clip(qr + seg_off + .8 * d_flu + caq + shock + rng.normal(0, .10))
        pers = clip(qp + seg_off + fit - .3 * L["sev"] * (1 if ai else .2) + caq + .8 * shock + rng.normal(0, .12))
        grammar = clip(readab + .1 + rng.normal(0, .12))
        cta = clip({"Manual": 4.3, "Prompt-only": 3.8, "RAG": 4.1, "Hybrid RAG": 4.2, "RAG+HITL": 4.5}[wf] + seg_off + shock + rng.normal(0, .15))
        compliance = clip(factual - .1 * (risk - 1) * (1 if ai and wf != "RAG+HITL" else .3) + rng.normal(0, .12))
        # multilingual rubric
        dq_all = (d_fact, d_flu, d_tone, d_cult)                   # language penalty for the holistic score
        mf, mfl, mt, mc = ML[wf]
        if wf == "RAG+HITL" and lang == "Yoruba":
            mf, mfl, mt, mc = YOR["fact"][si], YOR["flu"][si], YOR["tone"][si], YOR["cult"][si]
            d_fact = d_flu = d_tone = d_cult = 0.0; seg_off_ml = 0.0
        else:
            seg_off_ml = seg_off
        f_ret = clip(mf + seg_off_ml + d_fact + shock + rng.normal(0, .10))
        fluency = clip(mfl + seg_off_ml + d_flu + shock + rng.normal(0, .10))
        b_tone = clip(mt + seg_off_ml + d_tone + shock + rng.normal(0, .10))
        cult = clip(mc + seg_off_ml + d_cult + fit + shock + rng.normal(0, .12))
        ov_lat = pb["q"][si] + caq + shock + .45 * np.mean(dq_all) + .5 * fit + rng.normal(0, .05) + .02
        r = [float(np.clip(np.rint(2 * (ov_lat + b + rng.normal(0, .22))) / 2, 1, 5)) for b in rater_bias[mkt]]
        overall = float(np.mean(r))

        # ---------------- revisions, edits, acceptance ----------------
        rev_mean = (YOR["rev"][si] if (wf == "RAG+HITL" and lang == "Yoruba") else pb["rev"][si]) \
            * (1 + .15 * (L["sev"] if (ai and not (wf == "RAG+HITL" and lang == "Yoruba")) else 0)) * (1.15 if (peak and not ai) else 1) * (C["ae"] if ai else C["me"]) ** .25
        revisions = int(1 + rng.poisson(max(rev_mean - 1, .05)))
        emaj, emin, fpa = EDITS[wf]
        esc = (C["ae"] if ai else C["me"]) ** .5 * (1 + .3 * L["sev"] * ai)
        major = int(rng.poisson(emaj / 10 * esc)); minor = int(rng.poisson(emin / 10 * esc))
        lg = np.log(fpa / (1 - fpa)) + 1.5 * (ov_lat - pb["q"][si]) - .4 * L["sev"] * ai - (.3 if ct == "Blog Article" else 0)
        first_pass = int(rng.random() < 1 / (1 + np.exp(-lg)))

        # ---------------- errors (delivered-asset audit) ----------------
        emult = C["err"] * risk * (L["err"] if ai else 1) * ((1.3 if peak else 1) if not ai else 1)
        if mkt == "Nairobi" and cc == "C1" and ai and wf != "RAG+HITL":
            emult *= 1.15                                          # organic-certification (Kilimohai) claim risk
        errs = {k: int(rng.poisson(v / 10 * emult)) for k, v in zip(ETYPES, ERR[wf])}
        halluc = errs["unsupported_claims"] + errs["missing_citations"] + errs["incorrect_facts"]
        n_claims = int(1 + rng.poisson(C["claims"] - 1))
        bad_claims = min(n_claims, errs["unsupported_claims"] + errs["incorrect_facts"] + errs["unsupported_statistic"] + errs["outdated_claim"])
        sim = float(rng.beta(*{"Manual": (1.5, 18), "Prompt-only": (2, 14), "RAG": (2, 16), "Hybrid RAG": (2, 18), "RAG+HITL": (2, 22)}[wf]))
        distinct = float(np.clip(rng.normal({"Manual": .40, "Prompt-only": .55, "RAG": .53, "Hybrid RAG": .56, "RAG+HITL": .60}[wf] + fit / 2, .07), 0, 1))

        # ---------------- retrieval ----------------
        if wf in ("RAG", "Hybrid RAG", "RAG+HITL"):
            hyb = wf != "RAG"
            fail = (.08 + L["ret"] * .5) if hyb else (.20 + L["ret"])
            if rng.random() < fail:
                pos = [0, 0, 0]
            else:
                k = rng.choice([3, 2, 1], p=[.88, .11, .01] if hyb else [.82, .17, .01])
                pos = [1] * k + [0] * (3 - k)
                if k < 3 and rng.random() < .25:
                    rng.shuffle(pos)
            nr = sum(pos); kb = n_rel_kb if not hyb else (3 if rng.random() < .90 else 4)
            hit, prec, rec = int(nr > 0), nr / 3, min(1, nr / kb)
            rr = next((1 / (i + 1) for i, v in enumerate(pos) if v), 0.0)
            docs = int(pb["docs"][si] if ct == "Social Post" else rng.choice([4, 5]))
        else:
            hit = prec = rec = rr = np.nan; docs = 0

        # ---------------- cost & tokens ----------------
        if ai:
            api = pb["api"][si] * C["api"] * L["tok"] * (1 + .12 * (revisions - 2)) * ln(.10)
            api = max(api, .005)
            tin = int(api * .4 / PRICE_IN); tout = int(api * .6 / PRICE_OUT)
            latency = float({"Prompt-only": 2.6, "RAG": 3.0, "Hybrid RAG": 3.2, "RAG+HITL": 3.2}[wf] * M["lat"] * ln(.20))
        else:
            api = 0.0; tin = tout = 0; latency = np.nan
        active = total - (0 if not ai else latency / 60 * revisions)
        lab_u = active / 60 * M["rate"]                               # uniform-rate scheme (primary)
        lab_r = active / 60 * M["rate"] * (0.5 if ai else 1.0)        # report-rate scenario (AI Editor at 50% rate)
        platform = .03 if wf in ("RAG", "Hybrid RAG", "RAG+HITL") else 0
        ta = float(np.exp(rng.normal(np.log((9.0 if peak else 2.0) if not ai else (.7 if wf == "RAG+HITL" else .6) * (2.2 if peak else 1)), .35))) * C["md"] ** .3
        intervention = rng.choice(["Brand voice correction", "Factual verification", "Cultural localisation", "CTA optimisation", "Compliance review"],
                                  p=[.40, .25, .20, .10, .05] if not is_loc else [.30, .22, .33, .10, .05]) if wf == "RAG+HITL" else ""

        rows.append(dict(
            record_id=f"R{len(rows)+1:05d}", task_id=f"T{tid:05d}", workflow=wf, execution_order=int(order[wi]),
            brief_id=bid, segment=seg, client_code=cc, sector=sec, content_type=ct, nominal_words=C["words"],
            market=mkt, country=M["country"], language=lang, language_type="Local" if is_loc else "English",
            replicate=rep, week=week, peak_season=int(peak), pilot_anchor_cell=int(anchor),
            evidence_basis=("Pilot-anchored (Lagos English/Yoruba observed)" if mkt == "Lagos" else "Literature-extrapolated (Nairobi/Accra)")
                           + ("; ablation arm = assumption" if wf == "Hybrid RAG" else ""),
            editor_id=eds[wi], editor_ai_experience_months=ed["exp"], native_speaker_review=int(is_loc and wf == "RAG+HITL"),
            gen_or_draft_min=gen, editing_min=edit, total_min=total, revisions=revisions,
            major_edits=major, minor_edits=minor, first_pass_accepted=first_pass, turnaround_days=round(ta, 2),
            rater1_overall=r[0], rater2_overall=r[1], rater3_overall=r[2], overall_quality=round(overall, 3),
            q_brand_consistency=round(brand, 2), q_factual_accuracy=round(factual, 2), q_readability=round(readab, 2),
            q_personalisation=round(pers, 2), q_grammar=round(grammar, 2), q_cta=round(cta, 2), q_compliance=round(compliance, 2),
            ml_factual_retention=round(f_ret, 2), ml_fluency=round(fluency, 2), ml_brand_tone=round(b_tone, 2), ml_cultural_fit=round(cult, 2),
            segment_distinctiveness=round(distinct, 3),
            **{f"err_{k}": v for k, v in errs.items()}, hallucination_errors=halluc, total_errors=sum(errs.values()),
            factual_claims=n_claims, unsupported_or_incorrect_claims=bad_claims, supported_claim_rate=round(1 - bad_claims / n_claims, 4),
            max_similarity=round(sim, 3), copyright_flag=int(sim > .30),
            retrieved_docs=docs, hit_at_3=hit, precision_at_3=prec, recall_at_3=rec, reciprocal_rank=rr,
            input_tokens=tin, output_tokens=tout, latency_s=round(latency, 2) if ai else np.nan,
            api_cost_usd=round(api, 4), platform_cost_usd=platform, labour_rate_usd_h=M["rate"],
            labour_cost_uniform_usd=round(lab_u, 3), total_cost_uniform_usd=round(lab_u + api + platform, 3),
            labour_cost_report_rate_usd=round(lab_r, 3), total_cost_report_rate_usd=round(lab_r + api + platform, 3),
            hitl_primary_intervention=intervention,
            data_status="SYNTHETIC (simulated; pilot-anchored and literature-calibrated)",
        ))

df = pd.DataFrame(rows)
assert len(df) == 50000
df.to_csv("data/simulated_tasks_50000.csv", index=False)
print(df.shape)
a = df[(df.pilot_anchor_cell == 1) & (df.language == "English") & (df.peak_season == 0)]
print(a.groupby("workflow")[["gen_or_draft_min", "editing_min", "total_min", "revisions", "overall_quality", "q_brand_consistency",
                             "q_factual_accuracy", "q_readability", "q_personalisation", "hallucination_errors", "api_cost_usd",
                             "total_cost_uniform_usd", "total_cost_report_rate_usd", "first_pass_accepted"]].mean().round(2).T)
y = df[(df.pilot_anchor_cell == 1) & (df.language == "Yoruba") & (df.peak_season == 0) & (df.workflow == "RAG+HITL")]
print(y[["gen_or_draft_min", "editing_min", "total_min", "revisions", "ml_factual_retention", "ml_fluency", "ml_brand_tone", "ml_cultural_fit"]].mean().round(2))
e = df[(df.pilot_anchor_cell == 1) & (df.language == "English") & (df.peak_season == 0) & (df.workflow == "RAG+HITL")]
print(e[["ml_factual_retention", "ml_fluency", "ml_brand_tone", "ml_cultural_fit"]].mean().round(2))
print(df[df.language == "English"].groupby("workflow")[["hit_at_3", "precision_at_3", "recall_at_3", "reciprocal_rank"]].mean().round(3))
