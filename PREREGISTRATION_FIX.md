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

---

## Addendum: Condition D, evidence with reliability

**Date:** 14 November 2026

**Written before any condition D results were collected.** Conditions A–C had already been run, so D is a follow-up designed after seeing them: C moved Gemini's job-title pattern toward 2015 reality, but it largely copied the evidence table, including for president, where the table was wrong for 2015. Nothing above this line has been changed.

### A data note found while designing D

Before late 2013, LendingClub's `emp_title` field held the **employer's name** (for example "us army" or "walmart"), not the job title. The 12 titles have 0 loans in 2007–2011, 2 in 2012 and 2,070 in 2013 (2,067 of them in Q4). So the "2007–2014" evidence used in C, and here, actually comes from **June 2012 to December 2014**, almost all from Q4 2013 onward, about 15,500 loans. The filter was still "issued before 2015", so nothing from 2015 leaks in.

### Question

If the evidence also says how reliable each number is, and the AI is told to give less weight to uncertain or unstable evidence, does its job-title pattern get closer to 2015 reality than with the plain evidence (C)?

### Evidence D

The same 12 titles and the same gaps as C, computed from loans issued before 2015 only. Each line adds two things:

- **The gap's 95% interval:** a bootstrap of that title's loans, 2,000 resamples, fixed seed.
- **A stability flag:** "stable" if the gap has the same sign in both halves of the evidence period, otherwise "unstable". The halves are split at the median issue date: **June 2012 – June 2014** against **July – December 2014**, about 7,700 loans each. (The originally planned split, 2007–2011 against 2012–2014, is impossible, because 2007–2011 has no job titles.)

The prompt adds one instruction: **"Give less weight to evidence that is uncertain or marked unstable."** The exact text is in `prompts/evidence_reliability.txt`, committed with this addendum.

**Known in advance from the evidence itself:**
- **Mechanic** is flagged **unstable** and has a wide interval (−2.0 to +3.8), so D should give it less weight.
- **President** comes out **stable**, with an interval that barely excludes zero (−4.2 to −0.0). Its evidence still points the wrong way for 2015 (−2.1 before, +1.4 in 2015), so **D may still follow the wrong evidence for president.**
- The other 10 titles are flagged stable.

### Design

The same 600 loans and the same batch plan as A–C, with the **job title shown**, `gemini-3.5-flash-lite` with default settings, and the plain prompt plus the reliability evidence block. This is one run, `fix_D`, of 24 requests, scored with the same safety code.

### Main rule

**D beats C** only if the paired bootstrap interval for **distance D − C lies entirely below zero**. Distance is defined as in the main measure above: the mean absolute gap between the AI's title pattern and the 2015 real gaps, both centred, with loans resampled within each title 2,000 times on the same resamples as A–C. If the interval lies entirely above zero, D is reported as **worse than C**. Otherwise the result is **"no detectable difference"**.

### Also reported

- **D's slope and rank correlation** against 2015 reality, with intervals. D is labelled **"over-corrects"** if its slope interval lies entirely above 1.
- **The president and mechanic shifts:** D − A against C − A on the same loans, with intervals.
- **AUC** for D, with the paired differences D − C and D − A, next to the sub-grade on the same loans.
- **The money spread** for D − A (notebook 06 method), next to C − A and 2015 reality.

### Limitations stated in advance

D was designed after seeing A–C. It is one run, compared with a C run made earlier in the day, so model updates or run-to-run noise could affect the comparison; the model version reported in each reply is recorded and compared. With 12 titles and 50 loans each, a difference between C and D has to be fairly large to be detected.

---

## Clarification note (added 14 November 2026, after the results; the text above is unchanged)

Where the A–C sections above say the evidence comes from "loans issued 2007–2014", the filter was "issued before 2015". Because LendingClub's `emp_title` field held the employer's name before late 2013, the loans that actually carry these 12 job titles were issued **June 2012 to December 2014**, almost all from Q4 2013 onward (about 15,500 loans). No 2015 data was used, and the evidence values are unchanged. The condition D addendum describes the same finding; it was written before any D results.
