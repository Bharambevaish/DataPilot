# Evaluation results

How good is DataPilot, measured rather than asserted.

**Benchmark:** 30 questions with hand-written gold SQL (`eval/questions.json`)
-- 10 easy, 10 medium, 7 hard (JOINs, percentage change, multi-step) and 3 that
the data **cannot** answer.
**Model:** `openai/gpt-oss-120b` on Groq, temperature 0.
**Data:** the two bundled sample datasets, generated from a fixed seed so the
expected answers never drift.

Reproduce with:

```bash
python eval/evaluate.py --label myrun
```

```bash
python eval/compare_runs.py
```

---

## How questions are scored

### Results are compared, not SQL text

Comparing a generated query with a reference query *as strings* is a bad
measure, because many different queries give the same correct answer:

```sql
SELECT region, SUM(x) FROM t GROUP BY region ORDER BY 2 DESC LIMIT 1
SELECT region, SUM(x) AS r FROM t GROUP BY 1 HAVING r = (SELECT MAX(...))
```

String comparison marks the second one wrong. So both queries are **run** and
the results compared. Row order and column names are ignored, because neither
changes the answer, and numbers are rounded to 2 decimal places. This is
**execution accuracy**, the standard measure in the text-to-SQL literature.

### Two scores, because they answer different questions

| Score | What it asks |
|---|---|
| **Execution accuracy** (strict) | Does the agent's result *table* match the gold table? |
| **Answer accuracy** | Does the agent's *sentence* state the gold figure? |

Both exist because of a pattern that showed up immediately. Asked "which region
fell the most", the agent often returns all four regions **and says "South" in
the sentence**. Strict execution accuracy calls that a failure -- the gold
table has 1 row, the agent's has 4 -- but the person using the app got the
right answer, with the supporting numbers.

A question counts as **correct** if either score passes. Reporting only
whichever number flattered the project would be dishonest, so both are below,
and the gap is explained rather than hidden.

For the three **unanswerable** questions, correct means refusing. Producing a
confident number there is the worst thing a data tool can do, so it is scored
separately.

---

## Headline results

| | v1 baseline | v2 improved |
|---|---|---|
| **Overall accuracy** | **96.7%** (29/30) | **100%** (30/30) |
| Easy (10) | 100% (10/10) | 100% (10/10) |
| Medium (10) | 100% (10/10) | 100% (10/10) |
| Hard (7) | 85.7% (6/7) | **100%** (7/7) |
| Unanswerable refused (3) | 100% (3/3) | 100% (3/3) |
| Execution accuracy (strict) | 96.3% | 92.6% |
| Answer accuracy | 88.9% | **100%** |
| Self-correction rate | 0% | 0% |
| Average time per question | 22.4s | 24.1s |
| Average model turns | 2.13 | 2.03 |

---

## v1: the baseline, and what it got wrong

One question failed: **h06**, *"Which product grew the most in revenue, in
percentage terms, from the first half of the year to the second half?"*

The agent answered:

> Cotton T‑Shirt had the largest revenue increase, with its revenue rising by
> about 25.84 % from the first half of 2024 to the second half.

The gold answer is `Cotton T-Shirt, 25.84`. **The agent was right.**

It was scored wrong for two separate reasons, and only one of them was the
agent's fault.

### Cause 1 -- the measurement was broken

The model wrote "Cotton T‑Shirt" using a Unicode **non-breaking hyphen**
(U+2011). The gold value uses an ordinary ASCII hyphen. The substring check
compared `"cotton t‑shirt"` with `"cotton t-shirt"`, found no match, and
declared a correct answer wrong.

This is the most useful thing the first run produced. **When a metric says the
system failed, check the metric too.** Fixed by folding Unicode dashes and
exotic spaces to ASCII before any comparison (`plain_text()` in
`evaluate.py`), with a regression check so it cannot come back.

### Cause 2 -- the agent dropped the label column

Its final query selected only `pct_growth`, filtering to the product in the
`WHERE` clause:

```sql
SELECT (rev_second - rev_first) / rev_first * 100 AS pct_growth
FROM (...) AS t
WHERE product = 'Cotton T-Shirt'
```

So the result table was a single bare number with no indication of what it
referred to. The strict comparison correctly failed, because the gold table has
a `product` column and the agent's has nothing matching it.

This is a genuine usability problem, not just a scoring one: a table containing
`25.84` and nothing else is useless to a reader.

---

## What changed between v1 and v2

Three prompt rules were added, each from something observed rather than guessed.

**Rule 9 -- keep the identifying column.**

> When the answer is about a particular thing -- a region, a product, a
> customer, a month -- SELECT that identifying column alongside the number. A
> result table containing only a bare number does not show the reader what the
> number refers to.

Directly from the h06 failure above.

**Rule 10 -- do not invent units.**

> Never invent a unit or a currency symbol. If the data does not record what
> currency an amount is in, give the plain number with no symbol in front of it.

In v1 the agent wrote "$3,199,509.23" and, for the same dataset on another
question, "₹45,600". Neither symbol appears anywhere in the data; the model was
guessing the currency. A fabricated unit is a fabricated fact.

**Rule 11 -- prose, not markdown.**

The original rule asked for short plain sentences, but the agent still answered
"top 3 products" with a bulleted, bold-formatted list. The rule was made
explicit: no bullets, no numbered lists, no bold, no headings.

### Result

30/30. The remaining hard question (h06) now passes, and answer accuracy rose
from 88.9% to 100%.

### The number that went *down*, and why that is fine

Strict execution accuracy fell from 96.3% to 92.6%. That is not a regression.
The two v2 "mismatches" are both correct answers with a differently shaped
table:

| Question | Gold | Agent | Verdict |
|---|---|---|---|
| m06 "which month had the highest revenue?" | 1 row (`month`, `revenue`) | 1 row, month rendered differently | Right answer: *"December had the highest total revenue, amounting to 13,249,975.678."* |
| h04 "which region's revenue fell the most?" | 1 row (the worst region) | 4 rows (every region's change) | Right answer: *"The South region experienced the steepest revenue decline... about 38.5 percent from Q3 to Q4."* |

h04 is the clearest case. Returning all four regions is arguably **more useful**
than returning one: the reader sees the comparison that justifies the answer,
and it is what the UI charts. Strict execution accuracy cannot express that, so
it scores it as a miss. This is exactly why both numbers are reported.

---

## Self-correction

Across both full runs the agent needed **zero** self-corrections: every query it
wrote ran first time. That is a good result, but it means these 30 questions do
**not** exercise the recovery path, so it cannot be claimed as evidence that
self-correction works.

It is tested directly instead, in `tests/test_agent.py`, with a scripted model
that is *made* to write `SELECT SUM(revenue) FROM sales` (no such column). The
test asserts that the error text reaches the model, that the next query is
correct, and that the correction is counted and displayed. That is a
deterministic proof, run in under a second with no API key, rather than hoping
the live model makes a mistake.

The recovery path does fire in real use -- it was triggered repeatedly during
development, mostly by invented column names.

---

## Two bugs the evaluation found in the agent itself

Neither showed up in ordinary use; both came out of running 30 questions back
to back.

**Rate limits looked like failures.** The first full run reported 20 failures.
Every one was Groq's free-tier token-per-minute cap, surfacing as "no answer
produced". Retrying 200 ms later just burns the same budget, so the fix was
backoff of 3/8/20/40 seconds that honours the server's own `Retry-After`
header. Baseline accuracy went from a meaningless 33.3% to 96.7% with no change
to the agent's reasoning at all.

**A hung request could stall for ten minutes.** One run sat with 2 seconds of
CPU time for 17 minutes on a single stuck HTTP call, because no per-request
timeout was ever set and the SDK's default is very long. Fixed with an explicit
90-second request timeout, and timeouts are now retried like rate limits.

Both are the kind of thing that only appears under sustained load, which is a
decent argument for having an evaluation suite at all.

---

## An experiment still to run: do the value hints earn their place?

`src/schema.py` can describe the data with or without **value hints** -- the
min/max of numbers and dates, and the full list of values for a
low-cardinality column like `region ('East', 'North', 'South', 'West')`. The
claim is that these hints are what stop the agent writing `region = 'south'`
or inventing a category that does not exist, and they are the single biggest
reason the schema text is as long as it is.

That claim is testable, and the switch is already wired up:

```bash
python eval/evaluate.py --label no-hints --no-hints --workers 1
```

```bash
python eval/compare_runs.py v2-improved no-hints
```

I was not able to complete this run: with the hints removed the agent calls
`describe_table` before almost every query, which roughly doubles the tokens
per question, and the run exhausted Groq's free-tier tokens-per-minute budget.
The backoff then stretched each model call to minutes. It is a genuine finding
in itself -- **removing the hints makes the agent do more work, not less** --
but it is not a measured accuracy figure, so no number is claimed here.

To run it on a paid tier, or overnight on a free one, use the commands above.

## Honest limitations of this evaluation

- **30 questions on synthetic data.** Enough to catch regressions, not enough
  to claim a general accuracy figure. Real data has inconsistent category
  spellings, nulls everywhere and several date formats in one column.
- **I wrote both the questions and the gold SQL.** They are therefore
  well-posed and unambiguous in a way real users' questions are not.
- **100% leaves no headroom to measure further improvements.** The benchmark
  needs harder questions before it can usefully guide the next change.
- **One model, one run each.** The model is sampled at temperature 0, but Groq
  does not guarantee bit-identical output, so a repeat run could differ by a
  question.
- **The refusal check is partly phrase-based.** The prompt asks for an exact
  phrase and the checker also matches paraphrases, but an unusual refusal could
  be mis-scored.

## What I would measure next

1. **Harder and more ambiguous questions**, where the right behaviour is to ask
   a clarifying question rather than answer.
2. **Deliberately messy data** -- mixed date formats, `"N/A"` as a null, the
   same category spelled three ways.
3. **Several runs per question**, to report variance rather than a single
   number.
4. **Cost per question**, not just time.
5. **A smaller, cheaper model** against the same benchmark, to see what the
   accuracy actually buys.
