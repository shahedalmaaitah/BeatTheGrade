# Pre-registration: Can Evidence Cure the Blindness?

**Date:** 14 November 2026

**Written before any results were collected.** No AI model has scored any of the loans or prompts described here. The commit time of this file, together with the batch plan and the evidence prompt committed with it, is the proof.

## Background

In the job-title test (notebooks 04–05), Gemini 3.5 Flash-Lite essentially ignored job titles, even though within the same grade some titles really defaulted up to about 5 points more or less often. This experiment asks whether giving the AI real evidence changes that.

## Question

If the AI is given real evidence about job-title risk, do its estimates for each job move toward reality, and does it rank 2015 loans better?

## The evidence (no 2015 data)

For each of the 12 titles, the real gap is computed from **loans issued 2007–2014 only** (notebook 03): the default rate minus what the loan's sub-grade and year predicted. Nothing from 2015 leaks in. Each title gets one line in the prompt:

> Historical data: borrowers with this job title defaulted X percentage points more/less often than similar borrowers.

The exact wording is in `prompts/evidence.txt`, committed with this file. Two titles reverse sign between 2007–2014 and 2015 (president: −2.1 before, +1.4 in 2015; mechanic: +0.7 before, −0.8 in 2015). So for those two titles the evidence points the wrong way, as real historical evidence sometimes does.

## Test loans

**600 real 2015 loans** whose job title is exactly one of the 12 titles: 50 per title, drawn at random with a fixed seed, with every application field filled in. The 40 base applications and the 500 competence-check loans are excluded. These are real loans with real outcomes, not counterfactual copies. With about 15% defaulting, we expect about 90 defaults.

## Three conditions

- **A: no job title.** The job title field is blank.
- **B: job title shown.** The loan's real job title.
- **C: job title plus evidence.** As in B, with the 12-line evidence table added to the prompt.

Everything else is identical: `gemini-3.5-flash-lite` with default settings, the plain prompt (with the evidence block in C only), and the same fields and safety code as notebook 04. The batch plan is **24 batches of 25**, balanced by title (2–3 of each title per batch), random order, and it is **the same for A, B and C**, so every loan sits with the same neighbours in all three. That makes 72 requests and one run per condition. Run names are `fix_A`, `fix_B` and `fix_C`.

## Main measure: does C move the AI's title pattern toward reality?

For each condition, a title's **pattern value** is the AI's average estimate for that title's 50 loans minus its average over all 600. This is compared with the **2015 real gap** for each title (notebook 03).

- **Rank correlation** (Spearman) and **slope** of the AI's pattern against the 2015 real gaps, for A, B and C.
- **Mean absolute distance** between the AI's pattern and the 2015 gaps, with both centred on their average over the 12 titles.
- **95% bootstrap intervals** that resample loans within each title (50 per title), 2,000 times. The C − B and C − A differences are computed on the same resamples (paired).
- **The rule:** "C moves toward reality" **only if the paired interval for the distance C − B lies entirely below zero**, meaning C's title pattern is closer to 2015 reality than B's. If that interval lies entirely above zero, C is reported as "moves away from reality". Otherwise the result is "no detectable change".
- **Supporting measures:** the rank correlation and slope are reported alongside, but they do not decide the outcome. A bigger reaction is not automatically a better one: an AI that copies the evidence could raise its slope while moving further from reality.
- **"Over-corrects":** if C's slope interval lies entirely above 1, C is labelled "over-corrects", meaning it exaggerates the real gaps. This applies whatever the distance rule says.
- With only 12 titles, all of these measures are noisy, and that is stated.

## Second measure: ranking risk

**AUC** for A, B and C on the same 600 loans, next to the **sub-grade's AUC** on those loans. Intervals come from a paired bootstrap of loans, 2,000 times, and include the C − A, C − B and C − sub-grade differences. **Stated upfront:** with about 90 defaults, AUC intervals are wide, so only large differences (roughly 0.05 or more) can be detected. A non-significant difference is reported as "no detectable change", not as "no change". The notebook also reports the smallest difference the design could detect.

## Fairness check

Job titles can partly stand in for sex and other protected traits, because some jobs are mostly held by women or by men. LendingClub's data has no sex field, so this can't be measured directly here. **Condition C therefore trades fairness risk for accuracy**, and that is reported whatever the result.

The trade is made concrete with the notebook 06 money method. For each title we take the title-driven shift: the average of C − A on the same loans, and likewise B − A. We map each shift onto a typical C1 borrower, find the closest 2015 sub-grade, and compute the total interest on a \$10,000, 36-month loan against C1. The reported number is the dollar spread between the cheapest and the most expensive title under B, under C, and under the 2015 real gaps.

## Limitations stated in advance

The test covers one model and one prompt, one run per condition, 600 loans, about 90 defaults and 12 titles. The evidence is a summary table, not individual records. Any improvement could come from the AI copying the table rather than reasoning; the president and mechanic titles, where the evidence is wrong for 2015, help show which.
