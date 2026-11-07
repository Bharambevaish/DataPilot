# DataPilot -- interview notes

My own notes on this project: the decisions I made, why I made them, and the
things I would be asked about. Written so I can explain any file in the repo.

---

## 1. What is an AI agent, as opposed to a chatbot?

A chatbot gets a question and produces text in one shot. Everything it knows
has to already be in its context window, so when it does not know a number it
produces a plausible-looking one.

An agent is given a question **and a set of tools**, and runs in a loop:

```
question -> model -> "call run_sql(...)" -> run it -> result -> model -> ... -> answer
```

The model chooses which tool to call and with what arguments; my code runs the
tool and feeds the result back. The loop repeats until the model answers or
hits a step cap. In DataPilot that loop is `Agent.ask()` in `src/agent.py`,
about 60 lines.

Two things follow from the loop that I did not have to program separately:

* **Self-correction.** When a query fails, the error text is what the tool
  returns. The model reads "Referenced column 'revenue' not found", and writes
  a corrected query next turn.
* **Grounding.** Because the only route to a number is a tool call, and the
  prompt forbids numbers from anywhere else, answers are tied to the data.

## 2. How does tool calling actually work?

Nothing magic happens. I send the API a list of tool definitions in JSON
Schema (`TOOL_DEFINITIONS` in `src/tools.py`), describing each tool's name,
purpose and arguments. The model replies with either normal text or a
structured `tool_calls` block naming a tool and giving JSON arguments.

My code then:
1. parses those arguments (`json.loads`, with a fallback if the model emits
   malformed JSON -- it then gets an error back and retries),
2. runs the matching Python function,
3. appends the assistant's turn **and** a `role: "tool"` message carrying the
   result, keyed by `tool_call_id`,
4. sends the whole conversation again.

The model never executes anything. It emits a request; my code decides whether
to honour it. That separation is what makes the safety layers possible.

The four tools are deliberately few: `list_tables`, `describe_table`,
`run_sql`, `make_chart`. Every one returns a **string**, because a string is
all the model can read.

## 3. Why DuckDB?

* It runs SQL directly over local files with no server to install or
  administer -- important for a project someone clones and runs in two minutes.
* It is fast columnar OLAP, which is the right shape for analytics
  (`GROUP BY`, `SUM`), in-process with no network hop.
* It reads CSV and Parquet natively and speaks a PostgreSQL-like dialect, so
  the SQL the model writes is ordinary SQL.
* Crucially for me, **it can open a database file read-only**
  (`read_only=True`) and can have filesystem access switched off entirely
  (`enable_external_access=False`). That gives a hard safety boundary enforced
  by the engine, not by my code.

Why not SQLite? It is row-oriented and slower at aggregations, and it lacks the
analytics functions and the CSV handling. Why not pandas alone? Then the model
would have to write Python, which is far more dangerous to execute than SQL,
and much harder to check.

## 4. How do I stop the agent changing or deleting data?

Three independent layers. Any one would block most attacks; all three means a
bad query has to defeat a parser, a read-only file handle, and an engine-level
switch.

**Layer 1 -- parse the SQL and allowlist it** (`src/sql_guard.py`).
I parse the query with `sqlglot` into a syntax tree and require the whole
statement to be a query (`exp.Query`, which covers `SELECT`, `WITH ... SELECT`
and `UNION`). Everything else -- `INSERT`, `UPDATE`, `DELETE`, `DROP`,
`CREATE`, `ALTER`, `ATTACH`, `COPY`, `PRAGMA`, `SET`, `INSTALL` -- fails that
one check automatically.

> The key point to make here: this is an **allowlist, not a blocklist**.
> Searching the text for the word "DROP" is easy to defeat with comments or odd
> spacing, and it can never cover a statement type you did not think of. By
> requiring the parsed statement to *be* a SELECT, anything else is refused
> whether or not I anticipated it.

It also rejects multiple statements (which kills `SELECT 1; DROP TABLE x`),
blocks file- and network-reading functions by name *and by prefix*
(`read_*`, `write_*`, `glob`, ...), checks every table named actually exists,
and caps every query with a row limit.

**Layer 2 -- the connection cannot do damage** (`src/executor.py`).
The DuckDB connection is opened `read_only=True` and with
`enable_external_access=False`. I tested this layer **on its own**, with the
parser bypassed: `CREATE`, `DELETE`, `COPY ... TO`, `read_csv_auto('/etc/passwd')`,
`ATTACH` and `INSTALL` are all refused by DuckDB itself. So a bug in my parser
code is not enough to lose data.

**Layer 3 -- no query can run away.** Every query carries a row limit and a
wall-clock timeout, cancelled with DuckDB's `interrupt()`.

`tests/test_safety.py` checks all of this: 18 kinds of dangerous query refused,
6 kinds of safe query working, limits and timeout enforced.

*What I would add in production:* a per-user database file rather than a shared
one, query cost estimation before execution, and an audit log of every executed
query (I log them already, but not tamper-proof).

## 5. Why must every number come from a query?

This is the hallucination problem, and it is the single biggest risk in a tool
like this: a confident wrong number is worse than no answer, because nobody
checks it.

Three things enforce it:

1. **The prompt** states that every number in the answer must come from a
   `run_sql` result in that conversation, and that no query means no answer.
2. **The loop** gives the model no other way to obtain a number.
3. **The code checks afterwards.** In `Agent.ask()`, if the model returns a
   normal answer but never produced a query result, I discard the answer and
   tell the user I could not answer. A fabricated number is never shown as a
   result. `tests/test_agent.py` test 3 proves this with a scripted model that
   tries to answer "about 95 million" without querying.

The UI reinforces it: every answer displays the SQL and the result table, so
the number is always one click from being checked.

The related behaviour is **refusing**. If the data cannot answer the question,
the model is told to say the exact phrase "cannot be answered from this data".
All three unanswerable evaluation questions are refused correctly.

## 6. What is execution accuracy, and why not compare the SQL text?

Comparing generated SQL to reference SQL as *strings* is a bad measure, because
many different queries give the same correct answer:

```sql
SELECT region, SUM(x) FROM t GROUP BY region ORDER BY 2 DESC LIMIT 1
SELECT region, SUM(x) AS r FROM t GROUP BY 1 HAVING r = (SELECT MAX(...))
```

String comparison marks the second one wrong. **Execution accuracy** runs both
queries and compares the *results*: same rows, same values. Row order and
column names are ignored because they do not change the answer, and numbers are
rounded to 2 decimal places.

I report **two** numbers, because they answer different questions:

* **Execution accuracy (strict)** -- the agent's table matches the gold table.
  This is the standard research measure.
* **Answer accuracy** -- the agent's *sentence* states the gold figure.

The gap between them is real and interesting. Asked "which region fell the
most", the agent often returns all four regions and says "South" in the
sentence. Strict execution accuracy calls that a miss (1 gold row vs 4 agent
rows); the person using the app got the right answer. Reporting only the
flattering number would be dishonest, so I report both and explain the gap.

## 7. What went wrong, and what I learned

Three real bugs found by testing rather than by reading the code:

* **The chart was drawn from the wrong query.** `make_chart` used "the last
  result", but the agent often runs one more query *after* the one it wants to
  chart. The chart was then built from the wrong table and labelled with axes
  that were not even in it. Fixed by binding the chart to the most recent
  result that actually contains the requested columns, and returning an error
  (triggering another loop pass) when none does.
* **My own metric was wrong.** The first evaluation marked a correct answer as
  a failure because the model wrote "Cotton T‑Shirt" with a Unicode
  non-breaking hyphen and the gold answer had an ASCII one. The lesson: when a
  metric says the system failed, check the metric too.
* **Rate limits looked like bugs.** The first full evaluation "failed" 20
  questions, all of them Groq's token-per-minute cap. Added backoff that
  honours the server's `Retry-After`.

## 8. Known limitations (be honest about these)

* **One LLM, one language.** Accuracy depends on the model. A different Groq
  model may score differently; `MODEL_NAME` is configurable for exactly that
  reason.
* **The evaluation is on data I generated.** 30 questions on two synthetic
  datasets is enough to catch regressions, not enough to claim a general
  accuracy figure. Real messy data (inconsistent categories, nulls everywhere,
  five date formats in one column) is harder.
* **No conversation memory.** Each question starts a fresh conversation, so
  "and what about the North?" does not work. That is a deliberate simplicity
  trade-off: no cross-question contamination, but no follow-ups either.
* **Whole-file loading.** Files are read into memory, so the 10 MB cap is real.
  Pointing DuckDB straight at the file, or at a real database, would remove it.
* **Value hints do not scale.** Listing distinct values works for a column with
  12 cities. A column with 50,000 customer names only gets three examples, and
  filters on those are more error-prone.
* **The refusal check is partly phrase-matching.** I ask for an exact phrase and
  also pattern-match paraphrases. A model that refuses in an unusual way could
  be scored as a wrong answer.

## 9. What I would do next

1. **Follow-up questions** -- keep the last result in context so "and by city?"
   works.
2. **Few-shot examples in the prompt** -- the most common accuracy win in
   text-to-SQL, and cheap to add.
3. **Show confidence** -- flag answers where the agent needed several
   corrections, so a shakier answer looks shakier.
4. **A bigger, nastier evaluation set** including deliberately ambiguous
   questions, where the right behaviour is to ask a clarifying question rather
   than guess.
5. **Connect to a real database** with a read-only Postgres user, which is how
   this would actually be deployed.

## 10. Quick answers to likely questions

**"How long does a question take?"**
About 20-25 seconds on average across the evaluation set, mostly model latency;
the SQL itself runs in milliseconds.

**"What happens if the model writes invalid SQL?"**
The guard or DuckDB rejects it, the error text goes back to the model, and it
tries again, up to the 6-step cap. If it never succeeds, the user is told
honestly rather than shown a guess.

**"Why cap it at 6 steps?"**
Every step is an API call with the whole conversation attached, so cost and
latency grow quickly. Across the evaluation the agent averaged about 2 model
turns, so 6 is generous -- it is a safety net, not a normal limit.

**"Why temperature 0?"**
So the same question gives the same answer. Reproducibility matters more than
variety when the output is a number someone will act on.

**"What is the hardest part?"**
Making the model understand what the columns *mean*. "Revenue" is not a column
in the sales table -- it has to be derived from quantity, unit price and
discount. The schema description and prompt rules do that work, which is why
`src/schema.py` matters more than its size suggests.
