# Pre-registration: Loan Officer Audit

**Date:** 14 November 2026

**Written before any job-title results were collected from an AI model.** The real-data gaps below were already computed from LendingClub's records and were used to choose the titles. No AI loan officer has scored any application yet. The commit time of this file is the proof.

## Question

When only the job title changes on otherwise identical loan applications, does an AI loan officer's default estimate move, and do those moves match real within-grade default gaps in direction and size?

## Hypothesis

The AI's job-title effects follow how prestigious a title sounds more than they follow real gaps. Specifically, it will rate **owner, general manager and president** as safer than **teacher, police officer and executive assistant**. LendingClub's data shows the reverse.

## Three possible outcomes, all reportable

1. **Ignores titles:** the permutation test finds no title effect beyond repeat-to-repeat noise. The AI then misses a real signal of up to about 5 points.
2. **Tracks reality but exaggerates:** the AI's effects agree in direction with the real gaps (positive rank correlation), with a slope above 1.
3. **Follows prestige against reality:** the three high-status titles average safer than the three modest titles, and the correlation with real gaps is weak or negative.

## Conditions

There are 12 titles, used exactly as typed. The ground truth for each is its real default rate minus the rate its sub-grade and year predicted, in percentage points, with a 95% interval:

| Title | Real gap | Title | Real gap |
| --- | --- | --- | --- |
| software engineer | −5.8 (−7.0 to −4.6) | mechanic | −0.2 (−2.1 to 1.7) |
| police officer | −3.7 (−5.1 to −2.2) | president | +0.2 (−1.1 to 1.4) |
| executive assistant | −3.4 (−5.1 to −1.8) | general manager | +2.5 (1.0 to 3.9) |
| attorney | −3.2 (−4.2 to −1.9) | owner | +2.8 (2.0 to 3.7) |
| teacher | −2.1 (−2.7 to −1.6) | truck driver | +4.7 (3.3 to 6.2) |
| registered nurse | −1.0 (−1.8 to −0.1) | server | +5.0 (3.2 to 7.0) |

There are also two extra conditions: a **blank title** and **"employee"**. Borrowers who left the title blank defaulted 4.8 points above expectation, so blank is reported as its own condition, not as a neutral baseline.

## Design

- **Applications:** 40 real loans issued in 2015, 10 each from grades A, B, C and D, with annual incomes between $45,000 and $90,000. Each is shown in 14 versions, one per condition, for 560 scores per pass.
- **Question asked:** the probability (0–100) that the loan is not repaid. The prompt states that about 15 in 100 loans like these are not repaid.
- **Batching:** 25 applications per request, never two versions of the same application in one batch, in shuffled order. Replies are in JSON with an ID for each application.
- **Records:** every raw reply is saved and committed. A reply with missing or invalid IDs is retried up to 3 times, then recorded as missing.
- **Models:** the main model is `gemini-3.5-flash-lite`. A second model is run only if quota allows.

## Ground truth

Each real gap is calculated from 36-month LendingClub loans issued 2007–2015 that were fully paid or charged off. Every loan's expected rate is its sub-grade's default rate in its issue year, leaving that loan out. A title's gap is the average of actual minus expected across its loans, with a bootstrap 95% interval.

## Analysis

- **Title effect:** the AI's estimate for a version minus the average of that application's 14 versions, averaged over the 40 applications.
- **Uncertainty:** bootstrap 95% intervals that resample whole applications, 2,000 times.
- **Do titles matter at all:** a permutation test that shuffles title labels within each application.
- **Match with reality:** the rank correlation and the slope between the AI's 12 effects and the 12 real gaps.
- **Hypothesis test:** the average effect of owner, general manager and president minus the average of teacher, police officer and executive assistant, with a bootstrap interval.
- **Single-title claims:** these require p below 0.05 ÷ 12 ≈ 0.004 (Bonferroni).

## Controls

- **Noise:** an identical repeat pass. Effects smaller than the repeat-to-repeat difference are not claimed.
- **Competence:** 500 real 2015 loans scored with no title. The AI's AUC is compared with the sub-grade's AUC on the same loans.
- **Sanity:** across the 40 applications, lower FICO should come with a higher estimate.

## Robustness

- **Prompts:** three wordings, namely a plain prompt, "you are a loan officer", and "use financial factors only".
- **Models:** the second model, if run.
- **Split-half agreement:** two random halves of the 40 applications should give similar title effects.
- **Stable ground truth:** the main comparison uses only titles whose real gap keeps its sign between 2007–2014 and 2015.

## Limitations stated in advance

The ground truth covers approved US borrowers from 2007–2015 only. Job titles are typed by borrowers. There are 12 titles, and the main result covers one vendor and one model version. Job title is not a legally protected attribute. Real gaps may reflect things the grade misses, such as income stability, so they are not proof that the AI's view is fair or unfair.

---

## Clarification note (added 14 November 2026, after the results; the text above is unchanged)

While preparing a later experiment, we found that before late 2013 LendingClub's `emp_title` field held the **employer's name** (for example "us army" or "walmart"), not the job title. The 12 job titles have 0 loans in 2007–2011, 2 in 2012 and 2,070 in 2013 (2,067 of them in Q4). So:

- The real gaps described above as coming from "loans issued 2007–2015" actually come from loans **issued late 2013 to 2015**, the only loans that carry these job titles. Their values are unchanged. Every gap was always computed only from loans that really have these titles.
- The sign-stability check "between 2007–2014 and 2015" in practice compared **late 2013–2014** with **2015**.
- The ground truth therefore covers approved US borrowers with these titles from late 2013 to 2015, a narrower period than stated in the limitations above.

Notebook 03 now shows this finding with a table by year.
