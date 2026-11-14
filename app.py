"""LoanLens: check a personal-loan offer against what your real risk justifies.

Runs only on files committed to the repo (outputs/ and models/), so it works without data/.
Start it with:  python app.py   then open http://127.0.0.1:7860
"""
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

SUBGRADES = list(CUTOFFS.index)                                   # A1 ... G5, safest first
TITLES = [t for t in GAPS.index if GAPS.loc[t, "preregistered"]]  # the 12 tested job titles
OTHER = "other"
NAME_PERMUTATION_P = 0.86   # from notebook 07 (not stored in a CSV)
LOAN_MONTHS = 36
# Sub-grades with fewer than 200 loans in 2015 have default rates too noisy to match a job gap against
RELIABLE = RATES[RATES["loans"] >= 200]

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


def job_adjusted(subgrade, title):
    """Shift by the job's real gap, as in notebook 06: that sub-grade's 2015 default rate + gap -> closest sub-grade."""
    if title not in TITLES:
        return subgrade
    target = RATES.loc[subgrade, "default_rate_pct"] + GAPS.loc[title, "real_gap_pp"]
    return (RELIABLE["default_rate_pct"] - target).abs().idxmin()


def fair_subgrade(app, title):
    risk = model_risk(app)
    model_sg = deserved_subgrade(risk)
    return risk, model_sg, job_adjusted(model_sg, title)


def rate_of(subgrade):
    return float(RATES.loc[subgrade, "avg_int_rate_pct"])


def total_interest(rate_pct, principal, months=LOAN_MONTHS):
    r = rate_pct / 100 / 12
    payment = principal * r / (1 - (1 + r) ** -months)
    return payment * months - principal


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
def check_rate(title, offered_sg, *fields):
    app = build_app(*fields)
    loan = app["loan_amnt"]
    risk, model_sg, final_sg = fair_subgrade(app, title)
    offered_rate, fair_rate = rate_of(offered_sg), rate_of(final_sg)
    offered_int, fair_int = total_interest(offered_rate, loan), total_interest(fair_rate, loan)
    diff = offered_int - fair_int

    if title in TITLES:
        gap = GAPS.loc[title, "real_gap_pp"]
        word = "less" if gap < 0 else "more"
        moved = (f"which moves you from <b>{model_sg}</b> to <b>{final_sg}</b>." if final_sg != model_sg
                 else f"but that's too small to change your sub-grade, so you stay at <b>{final_sg}</b>.")
        job_html = (f"Borrowers with the job title <b>{nice_title(title)}</b> defaulted <b>{abs(gap):.1f} points {word}</b> "
                    f"often than their grade predicted ({int(GAPS.loc[title, 'loans']):,} LendingClub loans), {moved}")
    else:
        job_html = ("No job adjustment: we only have reliable repayment data for the 12 tested job titles, "
                    f"so your fair sub-grade stays at <b>{final_sg}</b>.")

    if final_sg == offered_sg or abs(diff) < 25:
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
  <div class="ll-step"><div class="ll-k">After your job's real gap</div><div class="ll-v">{final_sg}</div>
    <div class="ll-s">{'adjusted' if final_sg != model_sg else 'no change'}</div></div>
  <div class="ll-step"><div class="ll-k">Fair interest rate</div><div class="ll-v">{fair_rate:.2f}%</div>
    <div class="ll-s">offered {offered_sg}: {offered_rate:.2f}%</div></div>
</div>
<div class="ll-card"><p>{job_html}</p>
<table class="ll-table">
  <tr><th></th><th>Sub-grade</th><th>Rate</th><th>Total interest on {money(loan)}, {LOAN_MONTHS} months</th></tr>
  <tr><td>What you were offered</td><td>{offered_sg}</td><td>{offered_rate:.2f}%</td><td>{money(offered_int)}</td></tr>
  <tr><td>What your risk justifies</td><td>{final_sg}</td><td>{fair_rate:.2f}%</td><td>{money(fair_int)}</td></tr>
  <tr class="ll-total"><td>Difference</td><td></td><td>{offered_rate - fair_rate:+.2f} pts</td><td>{sign}{money(diff)}</td></tr>
</table></div>
<div class="ll-verdict ll-{tone}">{verdict}</div>
<p class="ll-note">Rates are LendingClub's 2015 averages for each sub-grade. Your sub-grade comes from ranking your risk
among 2015 borrowers, so it doesn't depend on the model's exact percentage.</p>"""


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


def better_rate(title, offered_sg, *fields):
    app = build_app(*fields)
    _, _, base_sg = fair_subgrade(app, title)
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
            _, _, new_sg = fair_subgrade(new, title)
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
def interval(low, high, digits=1):
    return f"95% interval {low:+.{digits}f} to {high:+.{digits}f}"


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
  <div class="ll-card ll-real"><div class="ll-k">Reality · {int(g['loans']):,} LendingClub loans</div>
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
                        offered_sg = gr.Dropdown(SUBGRADES, value="C1", label="Sub-grade you were offered",
                                                 info="A1 = safest, G5 = riskiest")
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
            gr.HTML("<div class='ll-headline'>Fair but blind: the AI ignores your name (good) and your job (a missed signal).</div>"
                    "<p class='ll-note'>We gave Gemini the same real loan applications many times, changing only the job "
                    "title or only the first name, and compared its reaction with how borrowers really repaid.</p>")
            ai_job = gr.Dropdown([(nice_title(t), t) for t in TITLES], value="software engineer", label="Pick a job")
            ai_out = gr.HTML()
            gr.HTML(name_summary())
            with gr.Row():
                with gr.Column():
                    chart("ai_vs_reality.png", "Orange: how each job really repaid vs its grade. Blue: how much the AI's "
                          "estimate moved for that job. The AI stays near zero while reality swings ±5 points.")
                with gr.Column():
                    chart("name_test.png", "Changing only the first name: every effect sits inside the ±0.3-point "
                          "noise band, and every interval crosses zero.")
            gr.HTML("<div class='ll-placeholder'><div class='ll-k'>Placeholder · to be filled in later</div>"
                    "<p><b>Can evidence fix it?</b> If we show the AI the real repayment data for each job, does it start "
                    "using the job signal correctly? Results will go here.</p></div>")

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

    check_btn.click(check_rate, [job, offered_sg] + FIELDS, rate_out)
    better_btn.click(better_rate, [job, offered_sg] + FIELDS, better_out)
    ai_job.change(ai_view, ai_job, ai_out)
    demo.load(check_rate, [job, offered_sg] + FIELDS, rate_out)
    demo.load(ai_view, ai_job, ai_out)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, server_name="127.0.0.1", server_port=7860)
