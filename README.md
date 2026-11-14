# Beat the Grade: Stereotype vs Reality

**Track:** Finance
**Team:** [names]

> Same loan. Different job title. Should the AI care?

*This README becomes the project write-up. Fill in every [bracket] with real results before submitting.*

## The problem

Lenders are starting to test AI chatbots for loan decisions. If an AI judges borrowers by their job title more than real repayment data justifies, people pay higher rates or get rejected because of a stereotype.

## What we built

1. **Beat the Grade:** a default model that competes with LendingClub's own risk grades on 2015 loans it never saw.
2. **Stereotype vs Reality:** an AI loan officer (Gemini) scores identical applications that differ only by job title, and we compare its assumptions with real default rates for those jobs.
3. **Demo:** enter a loan and a job title and see all three side by side.

## What we added

Our own idea. The new question is whether an AI loan officer's job-title assumptions match real outcomes across LendingClub's loans.

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

## Sources

- LendingClub loan data 2007–2018Q4 (Kaggle, uploaded by wordsforthewise)
- pandas, scikit-learn, matplotlib, Gradio, Google Gemini API
- AI tools used: [list every AI tool you used, as the rules require]

This project is a prototype and is not financial advice.
