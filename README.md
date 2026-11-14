# Beat the Grade: Stereotype vs Reality

**Track:** Finance
**Team:** [names]

> Same loan. Different job title. Should the AI care?

*This README becomes the project write-up. Fill in every [bracket] with real results before submitting.*

## The problem

Lenders are starting to test AI chatbots for loan decisions. If an AI judges borrowers by their job title more than real repayment data justifies, people pay higher rates or get rejected because of a stereotype.

## What we built

1. **Beat the Grade:** a default model that competes with LendingClub's own risk grades on 2015 loans it never saw.
2. **Stereotype vs Reality:** an AI loan officer (Gemini) scores identical applications that differ only by job title, and we compare its assumptions with real default rates for those jobs. Those rates come from the 40,006 loans issued late 2013–2015 that carry one of the 12 tested titles; before late 2013, LendingClub's job field held employer names, not job titles.
3. **Demo:** enter a loan and a job title and see all three side by side.

## What we added

Our own idea. The new question is whether an AI loan officer's job-title assumptions match real outcomes, using the 40,006 LendingClub loans issued late 2013–2015 that carry a real job title from our list (not all 618,584 loans).

## How it works

| Notebook | What it does |
| --- | --- |
| `01_data.ipynb` | Loads the LendingClub file, keeps finished 36-month loans from 2007–2015, cleans the columns |
| `02_model_vs_grade.ipynb` | Trains on 2007–2014, tests on 2015, compares our model with LendingClub's grades, runs the money test |

Data: LendingClub accepted loans 2007–2018 (Kaggle). We kept [n] finished 36-month loans issued 2007–2015.
Model: [model], using only information known when each loan was made.

## Results

- Ranking (2015 AUC): our model [X] vs LendingClub sub-grade [Y]
- Within-grade money test: safest half by our model defaulted [A]% vs [B]% for the whole grade
- Stereotype test: [to come]
- Where it fails: [one example and why]

## What we'd do next

[One or two concrete next steps]

## How to run it

See `SETUP.md`.

## Running LoanLens

LoanLens is a small web app for someone who has been offered a personal loan and wants to check it before signing. It runs on the files already in this repo (`outputs/` and `models/`), so you don't need the 1.6 GB LendingClub data.

From the project folder, in PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:7860** in your browser. Press Ctrl+C in PowerShell to stop it.

**The four tabs**

1. **Is my rate fair?** Enter your application, job title and the sub-grade you were offered. LoanLens shows our model's risk and the sub-grade that risk deserves. If your job is one of the 12 we tested, it then adjusts for how far our model's risk missed for that job, since the model never sees job titles. Finally it shows the fair interest rate and how much more or less total interest you'd pay on a 36-month loan.
2. **Will an AI judge me fairly?** For each job: how borrowers really repaid compared with how much Gemini's estimate moved, plus the name test. The job gaps come from the 40,006 loans issued late 2013–2015 with one of the 12 titles.
3. **What would get me a better rate?** Tries borrowing less, paying down credit cards and lowering debt-to-income, in 10% steps up to 50%. It reports the smallest change that moves you up at least one sub-grade, with the new rate and the interest saved.
4. **For lenders: audit report.** The report card for every loan officer we tested, plus the money charts.

**How the numbers are worked out**

- **Deserved sub-grade:** your risk is ranked against 2015 borrowers, and you get the sub-grade at the same rank (`outputs/deserved_subgrade_cutoffs.csv`, notebook 02).
- **Job adjustment:** our model's own miss for the title is added to your risk, which is then ranked again. The miss is the "model miss vs average" column in `outputs/real_gaps.csv` (`model_gap_vs_all_loans_pp`, notebook 03): how much more or less often 2015 borrowers with that title defaulted than our model predicted, compared with its average miss over all 2015 loans. It is the job-related risk our model doesn't already capture from income, debt and credit history. Titles not among the 12 get no adjustment.
- **Fair rate:** the average 2015 interest rate of that sub-grade. Total interest is for a standard 36-month loan with equal monthly payments.

**Files it reads**

- `models/default_model.joblib`
- From notebooks 02–07: `outputs/deserved_subgrade_cutoffs.csv`, `real_gaps.csv`, `ai_vs_reality.csv`, `name_test_effects.csv`, `name_test_contrasts.csv`, `report_card.csv`, `money_test.csv`, `auc_results.csv`, `grade_plus_model.csv` and four charts.
- Two small tables made once from 2015 loans in `data/loans_clean.parquet`:
  - `outputs/subgrade_rates_2015.csv`: loans, default rate and average interest rate per sub-grade.
  - `outputs/app_input_defaults.csv`: the typical 2015 value for each form field, used as defaults.

**Limits**

- **Prices are 2015 LendingClub averages**, not today's market rates.
- **The offer must be entered as a LendingClub-style sub-grade.**
- **Job numbers come from a small slice of the data.** LendingClub's job field held employer names before late 2013, so the 12 titles cover 40,006 loans issued late 2013–2015, not all 618,584. The model-miss adjustment uses only the 24,551 of those issued in 2015: from 644 loans (software engineer) to 6,820 (teacher). Adjustments for the smaller titles are less certain.
- **The model was saved with scikit-learn 1.9.1.** If an older version shows an `InconsistentVersionWarning`, run `pip install scikit-learn==1.9.1`.

LoanLens is a prototype for research and is not financial advice.

## Sources

- LendingClub loan data 2007–2018Q4 (Kaggle, uploaded by wordsforthewise)
- pandas, scikit-learn, matplotlib, Gradio, Google Gemini API
- AI tools used: [list every AI tool you used, as the rules require]

This project is a prototype and is not financial advice.
