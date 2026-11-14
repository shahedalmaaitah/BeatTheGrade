"""LoanLens: check a personal-loan offer against what your real risk justifies.

Runs only on files committed to the repo (outputs/ and models/), so it works without data/.
Start it with:  python app.py   then open http://127.0.0.1:7860
"""
import re
from html import escape
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, MODELS = ROOT / "outputs", ROOT / "models"

# ---------------------------------------------------------------- data the app needs
bundle = joblib.load(MODELS / "default_model.joblib")          # notebook 02's main model
MODEL, FEATURES = bundle["model"], bundle["features"]
CUTOFFS = pd.read_csv(OUT / "deserved_subgrade_cutoffs.csv", index_col=0)   # notebook 02, rank-based
RATES = pd.read_csv(OUT / "subgrade_rates_2015.csv", index_col="sub_grade")  # 2015 averages per sub-grade
GAPS = pd.read_csv(OUT / "real_gaps.csv", index_col="condition")             # notebook 03
AI = pd.read_csv(OUT / "ai_vs_reality.csv", index_col=0)                     # notebook 05
NAME_EFFECTS = pd.read_csv(OUT / "name_test_effects.csv", index_col=0)       # notebook 07
NAME_CONTRASTS = pd.read_csv(OUT / "name_test_contrasts.csv", index_col="contrast")
REPORT = pd.read_csv(OUT / "report_card.csv")                                # notebook 06
MONEY = pd.read_csv(OUT / "money_test.csv", index_col="grade")               # notebook 02
GRADE_PLUS = pd.read_csv(OUT / "grade_plus_model.csv", index_col=0)["value"]  # notebook 02
DEFAULTS = pd.read_csv(OUT / "app_input_defaults.csv", index_col="field")["default"]
FIX_PATTERNS = pd.read_csv(OUT / "fix_patterns.csv", index_col=0)                   # notebook 08
FIX_MEASURES = pd.read_csv(OUT / "fix_measures.csv", index_col="condition")         # notebook 08
FIX_MONEY = pd.read_csv(OUT / "fix_money.csv", index_col=0)                         # notebook 08
FIX_B, FIX_C = "B: job title", "C: title + evidence"
FIX_REAL, FIX_EVIDENCE = "reality (2015 gap)", "evidence shown in C (pre-2015 gap)"
MONEY_B, MONEY_C, MONEY_REAL = "B: title only", "C: title + evidence", "2015 reality"
# Condition D (evidence + reliability labels). Its per-job number is computed the notebook 08 way: the job's average
# estimate minus the average over all 600 loans (this reproduces B and C in fix_patterns.csv to 0.001).
FIX_D = "D: evidence + reliability"
FIX_D_MEASURES = pd.read_csv(OUT / "fix_D_measures.csv", index_col="condition")
FIX_D_MONEY = pd.read_csv(OUT / "fix_D_money.csv", index_col=0)
_scores_d = pd.read_csv(OUT / "fix_scores.csv")[["loan_id", "title"]].merge(
    pd.read_csv(OUT / "fix_scores_D.csv"), on="loan_id", how="inner", validate="one_to_one")
assert len(_scores_d) == 600 and _scores_d["D"].notna().all(), "condition D should score all 600 loans"
FIX_PATTERN_D = _scores_d.groupby("title")["D"].mean() - _scores_d["D"].mean()
# Same-loan change in Gemini's estimate when the title + evidence is shown, vs no title (notebook 08)
FIX_SHIFTS = pd.read_csv(OUT / "fix_D_shifts.csv", index_col=0)
_fix_scores = pd.read_csv(OUT / "fix_scores.csv")
ALL_LOANS_C_SHIFT = float((_fix_scores["C"] - _fix_scores["A"]).mean())   # the same change, averaged over all 600

SUBGRADES = list(CUTOFFS.index)                                   # A1 ... G5, safest first
TITLES = [t for t in GAPS.index if GAPS.loc[t, "preregistered"]]  # the 12 tested job titles
OTHER = "other"
NAME_PERMUTATION_P = 0.86   # from notebook 07 (not stored in a CSV)
LOAN_MONTHS = 36
# Before late 2013, LendingClub's emp_title held employer names, not job titles. So every job-title number here
# comes from the loans with one of the 12 titles, which were issued late 2013-2015 (not all 618,584 loans).
TITLE_LOANS = int(GAPS.loc[TITLES, "loans"].sum())                  # real gaps: late 2013-2015 loans
TITLE_LOANS_2015 = int(GAPS.loc[TITLES, "model_loans_2015"].sum())  # our model's miss: 2015 loans
TITLE_PERIOD = "late 2013–2015"
# 95% interval of the model miss we apply: notebook 03's interval for the title's raw model miss, shifted by the
# model's average miss over all 2015 loans (the shift that turns model_gap_pp into model_gap_vs_all_loans_pp).
_MISS_SHIFT = GAPS["model_gap_pp"] - GAPS["model_gap_vs_all_loans_pp"]
MISS_LOW, MISS_HIGH = GAPS["model_ci_low"] - _MISS_SHIFT, GAPS["model_ci_high"] - _MISS_SHIFT

HOME = [("Mortgage", "MORTGAGE"), ("Rent", "RENT"), ("Own outright", "OWN")]
PURPOSES = [("Debt consolidation", "debt_consolidation"), ("Pay off credit cards", "credit_card"),
            ("Home improvement", "home_improvement"), ("Major purchase", "major_purchase"),
            ("Medical", "medical"), ("Car", "car"), ("Small business", "small_business"),
            ("Moving", "moving"), ("Vacation", "vacation"), ("House", "house"),
            ("Renewable energy", "renewable_energy"), ("Other", "other")]


def d(field):
    return float(DEFAULTS[field])


# ---------------------------------------------------------------- core calculations
def model_risk(app):
    row = pd.DataFrame([{f: app[f] for f in FEATURES}])
    return float(MODEL.predict_proba(row)[:, 1][0])


def deserved_subgrade(risk):
    """Rank-based: the sub-grade at the same percentile of 2015 risk (notebook 02)."""
    ok = CUTOFFS.index[CUTOFFS["risk_up_to"] >= risk]
    return ok[0] if len(ok) else SUBGRADES[-1]


def model_miss(title):
    """Our model's own miss for this title, in points: 2015 default rate minus the model's risk, compared with
    its average miss over all 2015 loans (notebook 03, 'model_gap_vs_all_loans_pp'). This is the job-related
    risk our model doesn't already capture from income, debt and credit history. 0 for titles we didn't test."""
    return float(GAPS.loc[title, "model_gap_vs_all_loans_pp"]) if title in TITLES else 0.0


def miss_is_clear(title):
    """Only adjust for jobs whose model-miss interval excludes zero."""
    return title in TITLES and (MISS_LOW[title] > 0 or MISS_HIGH[title] < 0)


def fair_subgrade(app, title, adjust=False):
    """Returns the model's risk, the deserved sub-grade, the sub-grade after the (opt-in) job adjustment, and
    whether the adjustment was applied."""
    risk = model_risk(app)
    applied = bool(adjust) and miss_is_clear(title)
    adjusted = max(risk + model_miss(title) / 100, 0.0) if applied else risk
    return risk, deserved_subgrade(risk), deserved_subgrade(adjusted), applied


def rate_of(subgrade):
    return float(RATES.loc[subgrade, "avg_int_rate_pct"])


def total_interest(rate_pct, principal, months=LOAN_MONTHS):
    r = rate_pct / 100 / 12
    payment = principal * r / (1 - (1 + r) ** -months)
    return payment * months - principal


def interval(low, high, digits=1):
    return f"95% interval {low:+.{digits}f} to {high:+.{digits}f}"


def steps_between(a, b):
    return SUBGRADES.index(a) - SUBGRADES.index(b)


def money(x):
    return f"${abs(x):,.0f}"


def about(x):
    return f"${round(abs(x), -1):,.0f}"


def html_table(frame, cls="ll-table"):
    head = "".join(f"<th>{escape(str(c))}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in frame.itertuples(index=False))
    return f"<div class='ll-scroll'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def nice_title(t):
    return t.title() if t != OTHER else "Other / not listed"


def build_app(annual_inc, emp_years, home_ownership, fico, credit_history_years, open_acc, total_acc,
              revol_bal, revol_util, dti, delinq_2yrs, inq_last_6mths, pub_rec, loan_amnt, purpose):
    checks = [(annual_inc, 1_000, 10_000_000, "Annual income"), (loan_amnt, 1_000, 40_000, "Loan amount"),
              (fico, 300, 850, "Credit score"), (dti, 0, 100, "Debt-to-income"), (revol_util, 0, 150, "Revolving utilization"),
              (emp_years, 0, 50, "Years employed"), (credit_history_years, 0, 80, "Years of credit history")]
    for value, low, high, label in checks:
        if value is None or not (low <= value <= high):
            raise gr.Error(f"{label} should be between {low:,} and {high:,}.")
    for value, label in [(open_acc, "Open accounts"), (total_acc, "Total accounts"), (revol_bal, "Revolving balance"),
                         (delinq_2yrs, "Late payments"), (inq_last_6mths, "Credit checks"), (pub_rec, "Public records")]:
        if value is None or value < 0:
            raise gr.Error(f"{label} can't be negative.")
    return {"loan_amnt": loan_amnt, "fico": fico, "annual_inc": annual_inc, "dti": dti, "revol_bal": revol_bal,
            "revol_util": revol_util, "open_acc": open_acc, "total_acc": total_acc, "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths, "pub_rec": pub_rec, "emp_years": min(emp_years, 10),
            "credit_history_years": credit_history_years, "home_ownership": home_ownership, "purpose": purpose}


# ---------------------------------------------------------------- tab 1: is my rate fair?
def like_subgrade(rate_pct):
    """The sub-grade whose 2015 average interest rate is closest to this rate."""
    return (RATES["avg_int_rate_pct"] - rate_pct).abs().idxmin()


def check_rate(title, offered_rate, adjust, *fields):
    app = build_app(*fields)
    if offered_rate is None or not (1 <= offered_rate <= 40):
        raise gr.Error("Interest rate offered should be between 1% and 40%.")
    loan = app["loan_amnt"]
    risk, model_sg, final_sg, applied = fair_subgrade(app, title, adjust)
    fair_rate = rate_of(final_sg)
    offered_like = like_subgrade(offered_rate)
    offered_int, fair_int = total_interest(offered_rate, loan), total_interest(fair_rate, loan)
    diff = offered_int - fair_int

    miss = model_miss(title)
    word = "less" if miss < 0 else "more"
    if title not in TITLES:
        job_html, job_tile = ("No job adjustment: we only have reliable repayment data for the 12 tested job titles, "
                              f"so your fair sub-grade stays at <b>{final_sg}</b>."), "not available"
    elif not miss_is_clear(title):
        job_html, job_tile = (f"No job adjustment for <b>{nice_title(title)}</b>: our model's miss for this job "
                              f"({miss:+.1f} points, {interval(MISS_LOW[title], MISS_HIGH[title])}) could be zero, so we "
                              "don't use it."), "not used"
    elif not applied:
        job_html, job_tile = (f"Job adjustment is off. In 2015, <b>{nice_title(title)}s</b> defaulted {abs(miss):.1f} "
                              f"points {word} often than our model predicted; tick “Adjust for my job title” to include "
                              "it."), "off"
    else:
        moved = (f"which moves you from <b>{model_sg}</b> to <b>{final_sg}</b>." if final_sg != model_sg
                 else f"but that's too small to change your sub-grade, so you stay at <b>{final_sg}</b>.")
        job_html = (f"Our model never sees job titles. In 2015, borrowers with the job title <b>{nice_title(title)}</b> "
                    f"defaulted <b>{abs(miss):.1f} points {word}</b> often than our model predicted, compared with its "
                    f"average miss ({interval(MISS_LOW[title], MISS_HIGH[title])}; "
                    f"{int(GAPS.loc[title, 'model_loans_2015']):,} loans). We adjust your risk by that amount "
                    f"({risk:.1%} → {max(risk + miss / 100, 0):.1%}), {moved}")
        job_tile = "adjusted" if final_sg != model_sg else "no change"

    if abs(diff) < 25:
        verdict, tone = "Your rate looks about right for your real risk.", "neutral"
    elif diff > 0:
        verdict, tone = f"You may be paying about {about(diff)} more than your real risk justifies.", "bad"
    else:
        verdict, tone = f"Your offer looks about {about(diff)} cheaper than your real risk would justify.", "good"

    sign = "+" if diff > 0 else "−" if diff < 0 else ""
    return f"""
<div class="ll-steps">
  <div class="ll-step"><div class="ll-k">Our model's risk</div><div class="ll-v">{risk:.1%}</div>
    <div class="ll-s">chance of not repaying</div></div>
  <div class="ll-step"><div class="ll-k">Sub-grade you deserve</div><div class="ll-v">{model_sg}</div>
    <div class="ll-s">same risk rank among 2015 borrowers</div></div>
  <div class="ll-step"><div class="ll-k">Adjusted for your job</div><div class="ll-v">{final_sg}</div>
    <div class="ll-s">{job_tile}</div></div>
  <div class="ll-step"><div class="ll-k">Fair interest rate</div><div class="ll-v">{fair_rate:.2f}%</div>
    <div class="ll-s">you were offered {offered_rate:.2f}%</div></div>
</div>
<div class="ll-card"><p>{job_html}</p>
<table class="ll-table">
  <tr><th></th><th>Sub-grade</th><th>Rate</th><th>Total interest on {money(loan)}, {LOAN_MONTHS} months</th></tr>
  <tr><td>What you were offered</td><td>like {offered_like}</td><td>{offered_rate:.2f}%</td><td>{money(offered_int)}</td></tr>
  <tr><td>What your risk justifies</td><td>{final_sg}</td><td>{fair_rate:.2f}%</td><td>{money(fair_int)}</td></tr>
  <tr class="ll-total"><td>Difference</td><td></td><td>{offered_rate - fair_rate:+.2f} pts</td><td>{sign}{money(diff)}</td></tr>
</table></div>
<div class="ll-verdict ll-{tone}">{verdict}</div>
<p class="ll-note">The fair rate is LendingClub's 2015 average for your sub-grade; "like {offered_like}" is the sub-grade
whose 2015 average is closest to your offer. Market rates move over time, so treat the comparison as a guide. Your
sub-grade comes from ranking your risk among 2015 borrowers, so it doesn't depend on the model's exact percentage.</p>"""


# ---------------------------------------------------------------- tab 3: what would get me a better rate?
LEVERS = [
    ("Borrow less", "loan_amnt",
     lambda a, f: {"loan_amnt": a["loan_amnt"] * (1 - f)},
     lambda a, n: f"Borrow {money(n['loan_amnt'])} instead of {money(a['loan_amnt'])}"),
    ("Pay down credit cards", "revol_util",
     lambda a, f: {"revol_util": a["revol_util"] * (1 - f), "revol_bal": a["revol_bal"] * (1 - f)},
     lambda a, n: f"Card utilization {a['revol_util']:.0f}% → {n['revol_util']:.0f}%"),
    ("Lower debt-to-income", "dti",
     lambda a, f: {"dti": a["dti"] * (1 - f)},
     lambda a, n: f"Debt-to-income {a['dti']:.1f}% → {n['dti']:.1f}%"),
]
CUTS = [0.1, 0.2, 0.3, 0.4, 0.5]


def better_rate(title, offered_rate, adjust, *fields):
    app = build_app(*fields)
    _, _, base_sg, _ = fair_subgrade(app, title, adjust)
    base_rate = rate_of(base_sg)
    base_int = total_interest(base_rate, app["loan_amnt"])

    rows, best = [], None
    for name, key, change, describe in LEVERS:
        if app[key] <= 0:
            rows.append([name, "Already at 0", "", "", "", ""])
            continue
        found = None
        for cut in CUTS:
            new = {**app, **change(app, cut)}
            _, _, new_sg, _ = fair_subgrade(new, title, adjust)
            if steps_between(base_sg, new_sg) >= 1:
                found = (cut, new, new_sg)
                break
        if not found:
            rows.append([name, "Up to 50% less doesn't move you up", "", "", "", ""])
            continue
        cut, new, new_sg = found
        new_rate = rate_of(new_sg)
        new_int = total_interest(new_rate, new["loan_amnt"])
        from_rate = total_interest(base_rate, new["loan_amnt"]) - new_int
        rows.append([name, f"{describe(app, new)} ({cut:.0%} less)", f"{base_sg} → {new_sg}",
                     f"{new_rate:.2f}%", money(from_rate), money(base_int - new_int)])
        if best is None or (cut, -(base_int - new_int)) < (best[0], -best[5]):
            best = (cut, name, describe(app, new), new_sg, new_rate, base_int - new_int, from_rate)

    table = pd.DataFrame(rows, columns=["Change", "Smallest step that works", "Sub-grade", "New rate",
                                        "Saved from the better rate", "Total interest saved"])
    if best is None:
        head = (f"<div class='ll-verdict ll-neutral'>None of these changes (up to 50% less) moves you above "
                f"<b>{base_sg}</b>. Your other details are doing most of the work.</div>")
    else:
        cut, name, what, new_sg, new_rate, saved, from_rate = best
        extra = (f" ({money(from_rate)} of it from the better rate, the rest from borrowing less)"
                 if name == "Borrow less" else "")
        head = (f"<div class='ll-verdict ll-good'>Smallest change: <b>{what}</b>. That moves you from "
                f"<b>{base_sg}</b> to <b>{new_sg}</b>, a fair rate of <b>{new_rate:.2f}%</b> instead of {base_rate:.2f}%, "
                f"and saves about <b>{about(saved)}</b> in interest{extra}.</div>")
    note = (f"<p class='ll-note'>Each change is tried on its own, in steps of 10% up to 50%, using the details you "
            f"entered in “Is my rate fair?”. Savings are compared with your current fair rate ({base_sg}, {base_rate:.2f}%) "
            f"over {LOAN_MONTHS} months. Paying down cards lowers both your balance and your utilization.</p>")
    return head + "<div class='ll-card'>" + html_table(table, "ll-table ll-wrap") + "</div>" + note


# ---------------------------------------------------------------- tab 2: will an AI judge me fairly?


def ai_view(title):
    g, a = GAPS.loc[title], AI.loc[title]
    real_word = "less" if g["real_gap_pp"] < 0 else "more"
    real_sure = g["ci_low"] > 0 or g["ci_high"] < 0
    ai_word = "riskier" if a["ai_effect"] > 0 else "safer"
    if real_sure and not a["significant"]:
        verdict = (f"Reality says this job matters ({g['real_gap_pp']:+.1f} points). "
                   "The AI barely reacts: it misses a real signal.")
    elif real_sure and a["significant"]:
        verdict = "Both reality and the AI react to this job title."
    elif not real_sure and a["significant"]:
        verdict = "The AI reacts to this job title, but reality shows no clear gap: an unjustified reaction."
    else:
        verdict = "Neither reality nor the AI shows a clear effect for this job title."
    return f"""
<div class="ll-two">
  <div class="ll-card ll-real"><div class="ll-k">Reality · {int(g['loans']):,} loans, {TITLE_PERIOD}</div>
    <div class="ll-big">{g['real_gap_pp']:+.1f} pts</div>
    <p>{nice_title(title)}s defaulted {abs(g['real_gap_pp']):.1f} points {real_word} often than their grade predicted
    ({interval(g['ci_low'], g['ci_high'])}).</p></div>
  <div class="ll-card ll-ai"><div class="ll-k">Gemini 3.5 Flash-Lite · 40 applications × 2 runs</div>
    <div class="ll-big">{a['ai_effect']:+.2f} pts</div>
    <p>Changing only the job title to “{title}” made the AI rate the same applicant {abs(a['ai_effect']):.2f} points
    {ai_word} ({interval(a['ai_low'], a['ai_high'], 2)}).</p></div>
</div>
<div class="ll-verdict ll-neutral">{verdict}</div>"""


def name_summary():
    eff = NAME_EFFECTS.drop(index="(no name)", errors="ignore")
    biggest = eff["effect"].abs().max()
    sex, race = NAME_CONTRASTS.loc["Female - male"], NAME_CONTRASTS.loc["Black-sounding - white-sounding"]
    passed = not (sex["excludes_zero"] or race["excludes_zero"]) and NAME_PERMUTATION_P >= 0.05
    return f"""
<div class="ll-card"><div class="ll-k">The name test · same 40 applications, only the first name changed</div>
<p><b>{'The AI passes.' if passed else 'The AI does not pass.'}</b> No first name moved Gemini's estimate by more than
{biggest:.2f} points, and the names as a group made no difference (permutation p = {NAME_PERMUTATION_P:.2f}).</p>
<ul>
  <li>Female − male names: {sex['effect']:+.2f} points ({interval(sex['low'], sex['high'], 2)})</li>
  <li>Black-sounding − white-sounding names: {race['effect']:+.2f} points ({interval(race['low'], race['high'], 2)})</li>
</ul>
<p class="ll-note">The test could reliably detect a bias of about 0.3–0.4 points or more; smaller ones can't be ruled out.</p>
</div>"""


def first_number(text):
    return float(re.match(r"\s*([+-]?\d+(?:\.\d+)?)", str(text)).group(1))


def price(cell):
    """'B2 (-518 $)' -> ('B2', -518.0)"""
    m = re.match(r"\s*(\w+)\s*\(([+-]?\d+(?:\.\d+)?)\s*\$\)", str(cell))
    return m.group(1), float(m.group(2))


def spread(column):
    dollars = [price(v)[1] for v in FIX_MONEY[column]]
    return max(dollars) - min(dollars)


def evidence_summary():
    dist_b = first_number(FIX_MEASURES.loc[FIX_B, "distance"])
    dist_c = first_number(FIX_MEASURES.loc[FIX_C, "distance"])
    p, ps = FIX_PATTERNS.loc["president"], FIX_SHIFTS.loc["president"]
    return f"""
<div class="ll-two">
  <div class="ll-card"><div class="ll-k">Across all 12 jobs</div>
    <p>With evidence, Gemini's view of jobs got <b>about twice as close to reality</b>: its average distance from
    the real 2015 gaps fell from <b>{dist_b:.2f}</b> to <b>{dist_c:.2f}</b> points.</p>
    <p>Its <b>price spread between jobs</b> went from <b>${spread(MONEY_B):,.0f}</b> to <b>${spread(MONEY_C):,.0f}</b>,
    against <b>${spread(MONEY_REAL):,.0f}</b> in reality. That's the gap between the cheapest and the most expensive job
    for a typical C1 borrower, on $10,000 over 36 months.</p>
    <p>Adding reliability labels (D) left the distance about the same:
    <b>{first_number(FIX_D_MEASURES.loc[FIX_D, "distance"]):.2f}</b>, a change from C too small to detect.</p></div>
  <div class="ll-card ll-warn"><div class="ll-k">Honest warning</div>
    <p>Gemini <b>ended up exactly as accurate as the evidence table it was given, and no better.</b></p>
    <p>For president, the table was wrong: the 2012–2014 loans said presidents were <b>{p[FIX_EVIDENCE]:+.1f}</b>
    points (safer than their grade predicted), but in 2015 they were <b>{p[FIX_REAL]:+.1f}</b> points (riskier). On the
    same 50 president loans, adding the job title and the evidence lowered Gemini's estimate by
    <b>{abs(ps["C - A"]):.2f}</b> points (95% interval {ps["C - A interval"]}), the direction the wrong
    evidence pointed. Its estimates for all 600 loans fell by {abs(ALL_LOANS_C_SHIFT):.2f} points on average, so
    presidents didn't move further than other jobs, but nothing pulled them toward the truth either.</p>
    <p><b>An AI loan officer is only as good as its evidence.</b></p></div>
</div>
<p class="ll-note">Pricing by job title can partly stand in for sex and other protected traits, which this data can't
measure.</p>"""


PRESIDENT_LINE = ("<p class='ll-under'>Reliability labels didn't help here: the wrong number was marked stable, so "
                  "Gemini trusted it.</p>")


def evidence_view(title):
    row, money_row = FIX_PATTERNS.loc[title], FIX_MONEY.loc[title]
    b, c, r, ev = row[FIX_B], row[FIX_C], row[FIX_REAL], row[FIX_EVIDENCE]
    d = float(FIX_PATTERN_D[title])
    off_b, off_c, off_d = abs(b - r), abs(c - r), abs(d - r)

    def card(cls, label, value, sub, extra=""):
        return (f"<div class='ll-card {cls}'><div class='ll-k'>{label}</div><div class='ll-big'>{value:+.1f} pts</div>"
                f"<div class='ll-s'>{sub}</div>{extra}</div>")

    def priced(cell):
        sg, dollars = price(cell)
        cost = "same as C1" if dollars == 0 else f"{'+' if dollars > 0 else '−'}${abs(dollars):,.0f}"
        return f"<b>{sg}</b> ({cost})"

    name = nice_title(title) + "s"
    if off_c < off_b - 0.25:
        verdict, tone = (f"Evidence moved Gemini toward reality for {name}: it was {off_b:.1f} points off, now "
                         f"{off_c:.1f}."), "good"
    elif off_c > off_b + 0.25:
        verdict, tone = (f"Evidence moved Gemini further from reality for {name}: {off_b:.1f} → {off_c:.1f} points "
                         "off."), "bad"
    else:
        verdict, tone = f"Evidence barely changed Gemini's view of {name} ({off_b:.1f} → {off_c:.1f} points off).", "neutral"
    if np.sign(ev) != np.sign(r):
        verdict += (f" The evidence it was shown ({ev:+.1f}, from 2012–2014 loans) pointed the wrong way: in 2015 the "
                    f"real gap was {r:+.1f}.")
    elif off_c > off_b + 0.25:
        verdict += (f" It followed the evidence ({ev:+.1f}), which {'over' if abs(ev) > abs(r) else 'under'}stated "
                    f"this job's 2015 gap ({r:+.1f}).")
    verdict += f" With reliability labels too: {off_d:.1f} points off."

    return f"""
<div class="ll-four">
  {card("ll-fix-b", "Gemini alone", b, f"{off_b:.1f} pts from reality")}
  {card("ll-fix-c", "Gemini + LoanLens evidence", c, f"{off_c:.1f} pts from reality",
        f"<div class='ll-delta'>{c - b:+.1f} pts with evidence</div>")}
  {card("ll-real", "Reality, 2015 loans", r, "what really happened")}
  {card("ll-fix-d", "Gemini + evidence with reliability", d, f"{off_d:.1f} pts from reality",
        f"<div class='ll-delta ll-delta-d'>{d - c:+.1f} pts vs evidence alone</div>")}
</div>
{PRESIDENT_LINE if title == "president" else ""}
<div class="ll-verdict ll-{tone}">{verdict}</div>
<div class="ll-card"><div class="ll-k">What it would mean for a typical C1 borrower ($10,000, 36 months)</div>
  <p>Gemini alone: {priced(money_row[MONEY_B])} · Gemini + evidence: {priced(money_row[MONEY_C])} ·
  With reliability: {priced(FIX_D_MONEY.loc[title, FIX_D])} · Reality: {priced(money_row[MONEY_REAL])}</p>
  <p class="ll-s">These prices use only how much the job title itself moved Gemini's estimate (compared with the same
  loans shown without a title), so they can differ from the numbers above.</p></div>
<p class="ll-note">Points = how much riskier (+) or safer (−) Gemini rated this job than its typical loan, from 600 real
2015 loans (50 per job). The evidence shown to Gemini came only from loans issued before 2015. Reality is the job's 2015
default rate compared with what the grade predicted.</p>"""


# ---------------------------------------------------------------- tab 4: audit report
def stats_html():
    auc = pd.read_csv(OUT / "auc_results.csv", index_col=0).iloc[:, 0]
    tiles = [
        (f"{auc['Gradient boosting']:.3f}", "Our model's AUC on 2015 loans",
         f"LendingClub sub-grade: {auc['LendingClub sub-grade (baseline)']:.3f}"),
        (f"{GRADE_PLUS['auc_grade_plus_model']:.3f}", "Grade + our model",
         f"{GRADE_PLUS['difference']:+.3f} over the grade alone ({GRADE_PLUS['ci95_low']:+.3f} to {GRADE_PLUS['ci95_high']:+.3f})"),
        (f"{MONEY.loc['All grades', 'Default %: safest half']:.1f}% vs {MONEY.loc['All grades', 'Default %: riskiest half']:.1f}%",
         "Safest vs riskiest half", "default rate inside the same sub-grades"),
    ]
    return "<div class='ll-steps ll-three'>" + "".join(
        f"<div class='ll-step'><div class='ll-k'>{k}</div><div class='ll-v'>{v}</div><div class='ll-s'>{s}</div></div>"
        for v, k, s in tiles) + "</div>"


# ---------------------------------------------------------------- layout
CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.ll-hero { padding: 8px 2px 4px; }
.ll-hero h1 { font-size: 2rem; margin: 0; letter-spacing: -0.02em; }
.ll-hero p { margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 1.05rem; }
.ll-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 14px; }
.ll-three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.ll-step, .ll-card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 14px; padding: 14px 16px; }
.ll-k { font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: var(--body-text-color-subdued); }
.ll-v { font-size: 1.7rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.ll-s { font-size: .85rem; color: var(--body-text-color-subdued); }
.ll-big { font-size: 2.2rem; font-weight: 700; margin: 4px 0; font-variant-numeric: tabular-nums; }
.ll-card p { margin: 6px 0; line-height: 1.5; }
.ll-card ul { margin: 6px 0 6px 18px; }
.ll-two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.ll-four { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 4px; }
.ll-fix-b { border-top: 4px solid #a8a69f; }
.ll-fix-c { border-top: 4px solid #2a78d6; box-shadow: 0 0 0 2px rgba(42, 120, 214, .35); }
.ll-delta { display: inline-block; margin-top: 8px; padding: 2px 10px; border-radius: 999px; font-weight: 600;
  font-size: .85rem; background: rgba(42, 120, 214, .15); color: #2a78d6; }
.ll-fix-d { border-top: 4px solid #1f8a7a; }
.ll-delta-d { background: rgba(31, 138, 122, .15); color: #1f8a7a; }
.ll-under { margin: 10px 2px 0; font-size: 1rem; font-weight: 600; line-height: 1.45; }
.ll-warn { border-left: 6px solid #eb6834; }
.ll-section { font-size: 1.2rem; font-weight: 700; margin: 22px 0 2px; }
.ll-real { border-top: 4px solid #eb6834; }
.ll-ai { border-top: 4px solid #2a78d6; }
.ll-scroll { overflow-x: auto; }
.ll-table { width: 100%; border-collapse: collapse !important; margin-top: 10px; font-variant-numeric: tabular-nums;
  border: none !important; background: transparent !important; }
.ll-table th, .ll-table td { text-align: left; padding: 8px 6px !important; border: none !important;
  border-bottom: 1px solid var(--border-color-primary) !important; background: transparent !important;
  vertical-align: top; }
.ll-table tr, .ll-table thead, .ll-table tbody { border: none !important; background: transparent !important; }
.ll-table td:not(:first-child), .ll-table th { white-space: nowrap; }
.ll-report td, .ll-report th { white-space: normal !important; font-size: .9rem; }
.ll-wrap td, .ll-wrap th { white-space: normal !important; }
.ll-wrap td:first-child { font-weight: 600; }
.ll-report td:first-child { font-weight: 600; min-width: 150px; }
.ll-report td:last-child { min-width: 260px; }
.ll-table th { font-size: .8rem; color: var(--body-text-color-subdued); font-weight: 600; }
.ll-total td { font-weight: 700; border-bottom: none; }
.ll-verdict { margin: 14px 0 6px; padding: 16px 18px; border-radius: 14px; font-size: 1.2rem; font-weight: 600;
  line-height: 1.45; border-left: 6px solid var(--border-color-primary); background: var(--block-background-fill); }
.ll-bad { border-left-color: #eb6834; background: rgba(235, 104, 52, .10); }
.ll-good { border-left-color: #2a78d6; background: rgba(42, 120, 214, .10); }
.ll-neutral { border-left-color: #898781; }
.ll-note { font-size: .85rem; color: var(--body-text-color-subdued); margin-top: 8px; }
.ll-headline { font-size: 1.5rem; font-weight: 700; margin: 6px 0 2px; letter-spacing: -0.01em; }
.ll-placeholder { border: 2px dashed var(--border-color-primary); border-radius: 14px; padding: 18px;
  color: var(--body-text-color-subdued); }
.ll-placeholder b { color: var(--body-text-color); }
.ll-caption { font-size: .9rem; color: var(--body-text-color-subdued); margin: -4px 0 18px; }
.ll-footer { text-align: center; padding: 18px 0 6px; color: var(--body-text-color-subdued); font-size: .9rem;
  border-top: 1px solid var(--border-color-primary); margin-top: 18px; }
@media (max-width: 760px) {
  .ll-steps, .ll-three { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ll-two { grid-template-columns: 1fr; }
  .ll-four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def chart(path, caption):
    gr.Image(value=str(OUT / path), show_label=False, interactive=False, buttons=["fullscreen", "download"])
    gr.HTML(f"<p class='ll-caption'>{caption}</p>")


with gr.Blocks(title="LoanLens") as demo:
    gr.HTML("<div class='ll-hero'><h1>LoanLens</h1><p>Been offered a personal loan? Check it against your real "
            "risk before you sign.</p></div>")

    with gr.Tabs():
        # ---------------- tab 1
        with gr.Tab("Is my rate fair?"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=320):
                    gr.Markdown("#### Your loan offer")
                    with gr.Row():
                        loan_amnt = gr.Number(label="Loan amount ($)", value=d("loan_amnt"), precision=0, minimum=1000, maximum=40000)
                        offered_rate = gr.Number(label="Interest rate you were offered (%)",
                                                 value=round(rate_of("C1"), 2), minimum=1, maximum=40,
                                                 info="The yearly interest rate on your offer")
                    purpose = gr.Dropdown(PURPOSES, value=DEFAULTS["purpose"], label="What it's for")
                    gr.Markdown("#### About you")
                    with gr.Row():
                        annual_inc = gr.Number(label="Annual income ($)", value=d("annual_inc"), precision=0)
                        job = gr.Dropdown([(nice_title(t), t) for t in TITLES] + [(nice_title(OTHER), OTHER)],
                                          value="teacher", label="Job title")
                    with gr.Row():
                        emp_years = gr.Number(label="Years in current job", value=d("emp_years"), precision=0,
                                              info="10 means 10 or more")
                        home = gr.Dropdown(HOME, value=DEFAULTS["home_ownership"], label="Home")
                    adjust_job = gr.Checkbox(
                        value=False, label="Adjust for my job title",
                        info=f"Only used for jobs where our model's miss is clearly not zero "
                             f"({sum(miss_is_clear(t) for t in TITLES)} of the 12). Job titles can partly stand in "
                             "for sex, which lenders aren't allowed to use.")
                    gr.Markdown("#### Your credit")
                    fico = gr.Slider(600, 850, value=d("fico"), step=1, label="Credit score (FICO)")
                    with gr.Row():
                        dti = gr.Number(label="Debt-to-income (%)", value=d("dti"),
                                        info="Monthly debt payments ÷ monthly income, excluding mortgage")
                        revol_util = gr.Number(label="Credit-card utilization (%)", value=d("revol_util"),
                                               info="Card balances ÷ card limits")
                    with gr.Accordion("More credit details (optional, typical values filled in)", open=False):
                        with gr.Row():
                            revol_bal = gr.Number(label="Credit-card balances ($)", value=d("revol_bal"), precision=0)
                            credit_years = gr.Number(label="Years since first credit account", value=round(d("credit_history_years"), 1))
                        with gr.Row():
                            open_acc = gr.Number(label="Open credit accounts", value=d("open_acc"), precision=0)
                            total_acc = gr.Number(label="Total credit accounts ever", value=d("total_acc"), precision=0)
                        with gr.Row():
                            delinq = gr.Number(label="Late payments, last 2 years", value=d("delinq_2yrs"), precision=0)
                            inq = gr.Number(label="Credit checks, last 6 months", value=d("inq_last_6mths"), precision=0)
                            pub_rec = gr.Number(label="Public records (e.g. bankruptcy)", value=d("pub_rec"), precision=0)
                    check_btn = gr.Button("Check my rate", variant="primary", size="lg")
                with gr.Column(scale=7, min_width=360):
                    rate_out = gr.HTML()

        FIELDS = [annual_inc, emp_years, home, fico, credit_years, open_acc, total_acc,
                  revol_bal, revol_util, dti, delinq, inq, pub_rec, loan_amnt, purpose]

        # ---------------- tab 2
        with gr.Tab("Will an AI judge me fairly?"):
            gr.HTML("<div class='ll-headline'>The AI ignores your name (as the law requires) and your job (a missed "
                    "signal). Evidence moves it toward reality, but only as far as the evidence is right.</div>"
                    "<p class='ll-note'>We gave Gemini the same real loan applications many times, changing only the job "
                    "title or only the first name, and compared its reaction with how borrowers really repaid.</p>"
                    f"<p class='ll-note'>Where the job numbers come from: before late 2013, LendingClub's job field held "
                    f"employer names, not job titles. So each job's real repayment gap is measured on the "
                    f"{TITLE_LOANS:,} loans issued {TITLE_PERIOD} that carry one of these 12 titles, not on all "
                    f"618,584 loans.</p>")
            ai_job = gr.Dropdown([(nice_title(t), t) for t in TITLES], value="software engineer", label="Pick a job")
            ai_out = gr.HTML()
            gr.HTML("<div class='ll-section'>Can evidence fix it?</div>"
                    "<p class='ll-note'>A second test: Gemini scored 600 real 2015 loans with the job title shown, once "
                    "alone and once with a LoanLens table of each job's real repayment gap from earlier loans.</p>")
            fix_out = gr.HTML()
            gr.HTML(evidence_summary())
            gr.HTML("<div class='ll-section'>Does your name matter?</div>")
            gr.HTML(name_summary())
            with gr.Row():
                with gr.Column():
                    chart("ai_vs_reality.png", "Orange: how each job really repaid vs its grade. Blue: how much the AI's "
                          "estimate moved for that job. The AI stays near zero while reality swings ±5 points.")
                with gr.Column():
                    chart("name_test.png", "Changing only the first name: every effect sits inside the ±0.3-point "
                          "noise band, and every interval crosses zero.")

        # ---------------- tab 3
        with gr.Tab("What would get me a better rate?"):
            gr.Markdown("Uses the details from **Is my rate fair?**. We try a few simple changes and find the smallest "
                        "one that moves you up at least one sub-grade.")
            better_btn = gr.Button("Find my smallest change", variant="primary")
            better_out = gr.HTML()

        # ---------------- tab 4
        with gr.Tab("For lenders: audit report"):
            gr.HTML(stats_html())
            gr.Markdown("#### Report card: every loan officer we tested, on the same questions")
            gr.HTML("<div class='ll-card'>" + html_table(REPORT, "ll-table ll-report") + "</div>")
            with gr.Row():
                with gr.Column():
                    chart("money_test.png", "Inside every LendingClub grade, the half our model rated safest defaulted "
                          "less than the half it rated riskiest (2015 loans the model never saw).")
                with gr.Column():
                    chart("money_by_title.png", "If a typical C1 loan were priced to match each job's real risk: "
                          "the interest a borrower would pay more or less on $10,000 over 36 months.")

    gr.HTML("<div class='ll-footer'>Prototype for research. Not financial advice.</div>")

    check_btn.click(check_rate, [job, offered_rate, adjust_job] + FIELDS, rate_out)
    better_btn.click(better_rate, [job, offered_rate, adjust_job] + FIELDS, better_out)
    ai_job.change(ai_view, ai_job, ai_out)
    ai_job.change(evidence_view, ai_job, fix_out)
    demo.load(check_rate, [job, offered_rate, adjust_job] + FIELDS, rate_out)
    demo.load(ai_view, ai_job, ai_out)
    demo.load(evidence_view, ai_job, fix_out)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, server_name="127.0.0.1", server_port=7860)
