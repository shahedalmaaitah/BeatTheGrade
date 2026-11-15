# LoanLens: check your loan, and the AI judging it

## The problem

Lenders are starting to test AI chatbots for loan decisions. Before trusting one, we need to know two things. Does it treat people unfairly on traits the law forbids, like race or sex? And does it use the information that really predicts repayment?

## What we built

- **A default model** trained on 2007–2014 LendingClub loans and judged on 2015 loans it never saw.
- **Three pre-registered experiments** auditing Gemini as a loan officer on real applications.
- **LoanLens,** a four-tab app. It checks whether an offered rate is fair, shows how the AI would judge you, suggests what would get you a better rate, and gives lenders a report card.

## What we added

Most AI-bias tests only ask whether a protected trait changes the answer. We also compared the AI's reactions with **real repayment**. Using 40,006 real loans, we can tell whether ignoring a job title is fair or a missed signal. Then we tested whether evidence fixes it.

## How it works

- **The test versions:** the same real applications, in versions that differ only by job title (14 conditions) or first name (9).
- **Hiding the experiment:** the AI sees random IDs, and batches are balanced so no applicant appears twice in a batch.
- **Pre-registration:** every experiment was committed before its first request. The job-title and name tests were each run twice.
- **Statistics:** bootstrap intervals, permutation tests and Bonferroni correction.
- **The evidence test:** 600 real 2015 loans, scored with no title, with the title, and with the title plus each title's real default gap from loans issued before 2015.

## Results

- **Names:** no detectable effect, as the law requires. Female minus male: +0.16 points (−0.07 to +0.38). Black-sounding minus white-sounding: −0.10 (−0.37 to +0.16). Permutation p = 0.86.
- **Job titles:** within the same grade, real default ranged from 5.8 points below to 5.0 points above what the grade predicted. That's worth about \$800 in interest on a \$10,000 loan. Gemini moved less than 1 point for every title (slope +0.01).
- **Evidence:** the distance between the AI's job pattern and reality fell from 2.49 to 1.30. But the AI ended up exactly as accurate as the evidence it was given: the table alone scores 1.27 (exploratory), and the AI followed wrong evidence for president. Reliability labels changed nothing detectable (−0.04, interval −0.22 to +0.12).
- **Beating simple rules:**
  - Our model scored an AUC of 0.669, against 0.602 for FICO alone.
  - The grade plus our model scored 0.693, against 0.679 for the grade alone.
  - Borrowers our model places in a better sub-grade really defaulted less often than their grade predicted, by up to 4.5 points.

## What we'd do next

- **More AI models:** both second-model attempts (Gemma 4 31B and Gemini 3.8 Flash) were blocked by server errors.
- **The checks we didn't run:** the planned third prompt wording and split-half check.
- **The fairness cost:** measure it with data that records sex, since job titles can stand in for it.

**Limitations:** job titles exist only from late 2013; the data covers approved US borrowers only; and one AI model.

## Sources

- **Data:** LendingClub accepted loans 2007–2018Q4 (Kaggle, wordsforthewise).
- **Names study:** Bertrand & Mullainathan (2004), *American Economic Review* 94(4).
- **Models:** `gemini-3.5-flash-lite` (tested) and `gemma-4-31b-it` (attempted).
- **AI assistance:** Claude Opus 5.5 and Fable 5.2, through Claude Code and the Claude app.
- **Libraries:** pandas, scikit-learn, scipy, matplotlib, Gradio.

This project is a prototype and is not financial advice.
