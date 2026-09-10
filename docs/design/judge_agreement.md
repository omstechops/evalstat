# Design note — `judge_agreement()`

**Status:** interface only. Signature, docstring and assumptions are in
`src/evalstat/agreement.py`; the body raises `NotImplementedError`, the tests are
written next, and the function is not exported from the package namespace until
it works.

This note is the decision record. It says what was decided and why, what was
rejected, and what is still open. It does **not** repeat the assumption list:
that lives in the `judge_agreement` docstring, where it is read by whoever calls
the function, and a second copy here would be a second copy to keep in step.

---

## 1. What it answers

Can the LLM judge stand in for a human rater on this rubric, on this scale, on
these items?

The measurement is a weighted Cohen's kappa between the judge's labels and a
human's, with a confidence interval that resamples clusters rather than items,
and — when a second human's labels exist — the same coefficient between the two
humans, from the same resamples, with the signed difference between them.

The reason it is in this package at all is the same reason `paired_bootstrap` is:
`sklearn.metrics.cohen_kappa_score` returns the coefficient and stops. There is
no interval, and nothing anywhere that knows three rated items can come from one
recording. A judge validated with a point estimate has been validated against a
number whose sampling variation nobody looked at.

## 2. Scope

**In:** two raters, one ordered category set, complete pairing. Preference
judgements and absolute scores are the same object at different category counts.

**Out, deliberately:**

- **Krippendorff's alpha (D3).** Its two advantages over kappa are more than two
  raters and incomplete data. This study has two raters and complete data, so
  both advantages are paid for and neither is used. Deferred rather than
  rejected: a design with three raters or with dropped items is the case that
  makes it worth the second coefficient.
- **Gwet's AC1/AC2 (D5).** Rejected on the reason it would be added for. AC1
  exists largely because kappa falls under skewed marginals, so adding it *after*
  seeing a low kappa would be choosing the coefficient by its answer. A second
  coefficient beside the first also leaves the reader with "which one do I look
  at", and this package's whole subject is not leaving that question open. The
  diagnostics in section 5 address the same skew without producing a rival
  number.
- **PABAK (D6).** A rescaled `p_observed` presented as a new coefficient. That is
  the pattern this package exists to criticise, not to implement.
- **More than two raters.** Not a parameter that happens to be missing. At two
  raters the estimand is settled by naming a reference (§9.2); at three the
  question itself changes, and the coefficient for that design is the one
  deferred in D3.
- **Any reading of agreement as correctness.** Assumption 12. Against a fixed
  human label set the top-scoring judge is the one that reproduces the humans'
  mistakes best, and no argument inside the coefficient distinguishes that from
  competence.
- **A pass mark.** No threshold, no verbal band. The literature's labels for
  kappa ranges are convention with no sampling justification behind them; a
  study that needs a threshold sets one and pre-registers it.

## 3. One function, two rating tasks

Preference (A better / equal / B better) and absolute scores (1–5) are one
mathematics on a different number of ordered categories, so a second function
would be a second copy of the same body with a different docstring.

What makes one function safe is that `categories` is **required and ordered**
(D4). It is not inferred from the data, and the reason is a failure that is
silent and, worse, intermittent:

- a category nobody used disappears from an inferred set,
- the weight matrix loses a row and a column,
- and the scale is re-spaced, because linear weights count positions on it.

Re-spacing does not always move the coefficient, and that is the awkward part.
Normalised linear weights divide by `n_categories - 1`, so if the labels that
remain are still evenly spaced — a five-point scale used only at 1 to 4 — every
distance is multiplied by the same factor, the factor cancels between the
observed and expected sums, and kappa is unchanged. If what remains is unevenly
spaced — the same scale used only at 1, 2 and 5 — the distances move relative to
each other and kappa moves with them. Unweighted kappa is unaffected in both
cases.

So a project that infers its categories gets the right answer some of the time,
for a reason having nothing to do with the rating design, and no part of the
output distinguishes the two cases. The table and the prevalence index change
either way: the coefficient is reported on a four-point scale as though it were
the five-point scale the raters worked on. Requiring the set turns all of it
into an error at the call site.

The same requirement also carries the ordering, which is the other thing the
data cannot supply: `["equal", "A better", "B better"]` is a legal Python list
and a scale nobody rated on.

## 4. Weighting (D2)

**Linear, and `weights` is required with no default.**

Quadratic was rejected on what it does to adjacent disagreement. It charges
`(i - j)**2`, so a one-step disagreement costs a quarter of a two-step one and
is close to free. On a preference scale the one-step confusion — "equal" against
"A slightly better" — is exactly the distinction the study is trying to measure,
and a weighting that forgives it reports high agreement precisely where the
judge is least useful. Linear also leans less on assumption 1: it needs the
steps to be roughly equal, where quadratic needs their squares to mean
something.

No default, for the reason `rho` has no default in `power_analysis`: the only
available default is `unweighted`, and unweighted kappa on an ordered scale is
the specific error worth preventing. Being made to type the scheme is the point,
and it also means the scheme is present at the call site when someone reads the
call back.

`unweighted` stays available, because strict agreement is a real question and it
is the number other tools report by default. It is a choice the caller makes
visibly rather than one they fall into.

## 5. The chance model, and why the diagnostics are unconditional (D6)

`p_observed`, the prevalence index, the bias index and the cross-table are on
every result, not behind a flag.

Kappa is `(p_o - p_e) / (1 - p_e)`, and the same value arrives from tables that
should be read completely differently. Two failures in particular:

| What happened | What kappa does | What separates it |
|---|---|---|
| Both raters call almost everything "equal" | falls, while `p_o` stays high — the kappa paradox | prevalence index |
| The judge is systematically one step harsher | falls, though the two raters rank items identically | bias index |

Neither is visible in the coefficient, both are visible in the table, and a
report that quotes kappa alone leaves its reader to guess which case they have.
A systematically harsher judge is a **calibration** problem — recoverable by
moving the rubric anchors — and a judge disagreeing at random is not; that is a
difference worth more than the coefficient itself.

Definitions, which reduce to the standard 2×2 ones:

```
prevalence index  max_i p_ii - min_i p_ii
bias index        sum_i |p_i. - p_.i| / 2
```

At two categories the first is `|a - d| / n` and the second `|b - c| / n`, the
usual pair. Past two categories the bias index is the total variation distance
between the raters' marginals and has no real rival. The prevalence index is one
generalisation among several — the diagonal's variance, or its largest cell's
share, reduce to the same 2×2 definition and give different numbers past it —
and it is kept because it reduces to the standard definition, which is what
makes the figure comparable to a two-category one from the literature. §9.1
records the decision, and the docstring says in as many words that this is a
generalisation rather than the canonical quantity, so that nobody reads it as
one.

**Both are summaries, and the table is the evidence.** The indices exist to say
which of the two failures in the table above is present, so that a reader knows
what to look at; where the diagnosis carries weight, the answer is in `table`,
which is returned unconditionally for exactly this reason. An index disagreeing
with the table is the index being a summary, not the table being wrong.

**PABAK is not among them.** Rescaling `p_o` and calling it a chance-corrected
coefficient answers the paradox by removing the chance correction. If the
correction is wrong for this table, the table is what should be shown.

## 6. The human ceiling (D7)

### The argument the interface is built on

A judge–human kappa is **not** an upper bound, and it is **not** an unbiased
estimate of anything. It is biased upward by an unknown amount.

Kappa's chance model treats the two raters as independent given their marginals:
whatever they match on beyond those base rates is counted as agreement. A judge
trained on human preference data is not independent of a human in that sense.
The same items look hard to both, the same phrasing flatters both, the same
rubric wording is read the same way by both. That shared component is not chance
under the model, so it enters the numerator as agreement — and nothing in the
table separates it from the agreement the study is trying to establish. It is
bias, not variance, so a larger sample does not reduce it and the confidence
interval does not cover it.

What follows is not a correction — there is nothing to correct with — but a
comparison. The human–human kappa on the same items, under the same scale and
weighting, is the only available statement of how much agreement this rating
task admits at all. A judge at 0.71 is close to the ceiling when two humans reach
0.74 and nowhere near it when they reach 0.90, and the judge's own number cannot
tell those apart.

**This is why the ceiling comparison is a requirement of the design and not a
refinement of it**, and why the pre-registration carries the sentence in its own
words: *the judge–human kappa is not an upper bound; because the judge and the
human are not independent, it is an upward-biased estimate of unknown size.* It
connects to the circular-contamination rule in `CLAUDE.md` from the other end:
the human labels have to be produced without sight of the judge's, or the second
measurement is not a second measurement.

### Why an argument and not a second function

Two reasons, and the first is binding.

**The two kappas have to come from the same resamples.** They are computed on
the same items, so they are correlated, and the difference between them is a
paired quantity. Building it from two separately drawn intervals gives an
interval that is too wide, in a way nobody downstream would notice. A separate
`human_ceiling()` function would make the shared draw impossible to express
without a third function to coordinate them; `ceiling=` makes it structural.

**The difference is reported, not the ratio.** `kappa_judge / kappa_human` reads
naturally — "the judge achieves 95% of human agreement" — and is unstable
exactly where the study is interesting: the denominator is a small number
estimated with error, and a ratio of two noisy small numbers has a long tail.
The difference has none of that behaviour and its interval means what it says.

### What the body will have to settle

Two consequences of the paired requirement, both left to the implementation
stage and both already pinned by J12 as far as a test can pin them.

The shared resampler's statistic contract returns **one** number per resample,
and the ceiling comparison needs three: the judge's coefficient, the human's,
and their difference. So the body either makes a second aligned pass over the
same draws or the resampler grows a vector-valued variant. That is an
implementation choice; what is not a choice is that the draws be the same ones,
which is what J12's narrowness assertion tests.

The other is what to do with a resample in which one of the two coefficients is
undefined. Discarding it from the difference is forced. Whether it is also
discarded from the judge's own interval — which would make the judge's figure
depend on whether a ceiling was asked for — is open. J12 asserts the invariance
on a run where nothing was discarded, so the test pins the draws without
pre-empting the rule.

## 7. Degenerate tables (D8)

Kappa's denominator `1 - p_e` vanishes when a rater is constant: the chance model
then expects perfect agreement, and the coefficient is 0/0.

Two cases, deliberately handled differently:

| | Case | Behaviour | Why |
|---|---|---|---|
| Observed table | a rater used one category for every item | `ValueError`, naming the state | This is a fact about the data — a rubric dimension nobody discriminated on, or a judge that answered the same thing every time. It is not a numerical accident, and returning `nan` invites being read and reported as a number. |
| A resample | the draw left a rater constant | `nan`, discarded by the shared machinery | Expected at small cluster counts, counted in `n_valid`, surfaced as `DegenerateResampleWarning`, and an error past `MIN_VALID_FRACTION`. That policy already exists and is not duplicated here. |

The error message names the state rather than the arithmetic: which rater was
constant and which category they used, not "denominator is zero".

## 8. Decisions taken

| | Decision | Rationale |
|---|---|---|
| D1 | The cluster resampler is shared, keyed on item indices | Landed separately in `_resample.py`. What was ruled out in `paired_bootstrap` was hosting two *statistic contracts* in one function, not sharing the draw. Kappa cannot use the paired contract at all: `x - y` destroys the information, since (1, 2) and (4, 5) both give −1 and contribute differently to expected agreement. The statistic here closes over both label vectors and indexes each. |
| D2 | Linear weights; `weights` required, no default | Section 4. |
| D3 | Krippendorff's alpha deferred | Section 2. |
| D4 | One function; `categories` required and ordered | Section 3. |
| D5 | Gwet's AC1/AC2 out of scope | Section 2. |
| D6 | `p_observed`, PI, BI and the table reported unconditionally | Section 5. |
| D7 | The ceiling is an argument, reporting a difference from shared resamples | Section 6. |
| D8 | A degenerate observed table raises | Section 7. |
| D9 | `scikit-learn` as a dev dependency, as a test oracle only | Section 11. |
| — | Rows of `table` are the judge, columns the human | An orientation convention is needed and either one works; what matters is that it is stated where the array is documented, as `paired_bootstrap` states its sign convention. |
| — | `weights` and `categories` are carried on the result | A weighted kappa is not comparable to an unweighted one or to one on a different scale. A number that travels without them can be compared to anything, which is worse than not being comparable. |
| — | The prevalence index is `max_i p_ii - min_i p_ii` | §9.1. It reduces to the standard 2×2 definition, which is what keeps it comparable to a two-category figure from the literature; the docstring says it is a generalisation and not the canonical quantity. |
| — | The estimand names one human as the reference; `ceiling` is the ceiling, not a second reference | §9.2. It follows from the question being asked, and a question is not a default. |

## 9. Two questions, and how they were settled

Both were raised as open at the interface review and both were closed there.
They are kept in the note because a decision that leaves no trace is
indistinguishable later from an oversight.

### 9.1 The prevalence index past 2×2

**Kept as `max_i p_ii - min_i p_ii`.**

The alternatives — the variance of the diagonal, the largest diagonal cell's
share — reduce to the same 2×2 definition and give different numbers past it, so
none of them is canonical and choosing among them is choosing among conventions.
What settles it is comparability: this one reduces to the standard `|a - d| / n`,
so a figure reported here can be read beside a two-category figure from the
literature. Two conditions came with the decision and are implemented:

1. The docstring states in as many words that this is **a** generalisation of the
   2×2 definition, that other generalisations exist, and that they give different
   numbers. Without that sentence a reader takes it for the canonical quantity.
2. The docstring sends the reader to `table` when the diagnosis matters. The
   index is a summary; the cross-tabulation is the evidence, and it is returned
   unconditionally (D6) precisely so that the summary never has to be trusted on
   its own.

It affects a diagnostic and not the coefficient, and it is not worth more time
than that.

### 9.2 Which pairing, when a second human exists

**The asymmetric structure is kept: the judge is scored against `human`, and
`ceiling` is the second human, used only to measure the ceiling.**

This is an estimand decision, and it follows from the question the study asks —
*can the judge rate in my place?* The reference is therefore one named person,
the study owner, and not a pool. The second human exists to say how much
agreement the task admits at all, which is what a ceiling is for.

Averaging the judge's kappa over both humans answers a different question —
*does the judge rate like a typical human?* — which is more symmetric and is not
this study's question, because the person the judge would stand in for is the
owner. There is also a data reason not to average: the second human's labels
exist on the 40-item subset, not on every item, so an average over both raters
would be an average over an asymmetric sample and would not mean what its name
suggests.

**Reopened when a third rater arrives.** At three raters the question genuinely
changes — a reference person and a pool stop being the same thing to argue about
— and that is the design D3 defers Krippendorff's alpha to. Until then this is
settled, not deferred.

## 10. Test plan

Every case pins the function against an answer known independently of it. The
oracle rule follows D9: no coefficient value copied out of a paper, and every
reference figure has **two witnesses** — a table small enough to work through by
hand, whose `p_o` and `p_e` are written into the test as literals in the style of
`Z_0975` in `test_power.py` with the arithmetic in the test's docstring, and
`sklearn` computing the same table.

The suite lands in two stages and in two files, because the failure modes do not
mix. J1–J10, in `tests/test_agreement.py`, are the coefficient's arithmetic:
deterministic, fast, and wrong only if the mathematics is wrong. J11 and J12, in
`tests/test_agreement_interval.py`, are the clustered interval: stochastic, slow,
and dependent on the shared resampler. Written in one go, a red J11 would not
separate "the coefficient is wrong" from "the interval is wrong".

**J11's tolerances are provisional and say so in the file.** They are predictions
about a body that does not exist yet, and a tolerance in this project is measured
before it is defended: check whether the estimator is biased, at high precision,
and only then measure the spread at the setting the test fixes. When the body
lands, the coverage figures go into the `evalstat.agreement` module docstring the
way the paired bootstrap's table did, and J11's bounds are set from that
measurement rather than from this paragraph.

### Stage one — the coefficient (J1–J10)

| # | Case | Where the known answer comes from | Speed |
|---|---|---|---|
| J1 | Unweighted kappa, 2×2 | A hand-worked table: `p_o = 0.70`, `p_e = 0.50`, `kappa = 0.40`, each a literal, plus `sklearn.metrics.cohen_kappa_score` on the same data. | fast |
| J2 | Linear-weighted kappa, 3×3 and 5×5 | Two more hand-worked tables, against `cohen_kappa_score(weights="linear")`. The 5×5 is symmetric, so it pins a zero bias index and a zero prevalence index at the same time. | fast |
| J3 | Fixed points | Exact agreement is exactly 1.0 under either scheme. A table equal to the outer product of its own marginals is exactly 0.0 — the chance model reproduced exactly. Equality, not approximation. | fast |
| J4 | Invariances | Swapping the two raters leaves kappa, both indices and the transposed table unchanged. Relabelling the categories without reordering them changes nothing at all. Both are properties of the definition, checkable without an oracle. | fast |
| J5 | Prevalence: the kappa paradox | Hand-worked: `p_o = 0.90` with `kappa = 0.08 / 0.18`, prevalence index 0.80, bias index exactly 0. Pins §5's first row as arithmetic rather than as a claim in a docstring. | fast |
| J6 | Bias: a harsher judge | Hand-worked: `p_o = 0.65` with `kappa = 0.18 / 0.53`, prevalence index 0.05, bias index 0.25. The mirror image of J5, so that the two indices are shown to separate the two failures rather than both tracking kappa. | fast |
| J7 | Categories are not inferred | One five-point dataset used only at 1, 2 and 5. Dropping the unused labels changes the coefficient; dropping a label from the end of a dataset used at 1–4 does not, and the table and prevalence index change in both cases. Pins §3's *intermittent* failure in both of its states. Reordering the labels changes the coefficient too. | fast |
| J8 | The schemes are not comparable | A table whose disagreements are all adjacent: hand-worked `kappa` of `0.4140625 / 0.6640625` unweighted against `0.3046875 / 0.4296875` linear, so linear is the higher of the two, and neither number means anything without its scheme. | fast |
| J9 | Degenerate observed table | A rater who used one category throughout: `ValueError`, and the message names that rater and that category rather than the arithmetic. Both-constant is the same case. | fast |
| J10 | Input rejection | Mismatched lengths across `judge`, `human`, `ceiling` and `cluster`; a label absent from `categories`; `categories` repeating a label or holding one; a missing or unknown `weights`, `"quadratic"` included; the resampling arguments; fewer than two clusters. | fast |

### Stage two — the clustered interval (J11–J12)

| # | Case | Where the known answer comes from | Speed |
|---|---|---|---|
| J11 | Clustering restores coverage | The measurement `paired_bootstrap` already carries, repeated for kappa. Ratings are generated with a cluster-level shared component; the population kappa of that generator is computed by `sklearn` on a very large draw from it, never by this code; coverage of nominal 95% intervals is counted with and without `cluster=`, with its Monte Carlo error. The same test carries provenance: the seed is recorded when none was passed, a rerun at that seed reproduces the interval, and the few-clusters warning fires where it should. | slow |
| J12 | The ceiling comparison is paired | At a fixed seed, `difference` equals `kappa - ceiling.kappa` exactly, and its interval is **narrower** than one built from the two kappas' intervals as though they were independent. That inequality is D7's first reason and it is measurable. Passing `ceiling` also leaves the judge's own kappa and interval untouched at the same seed: the ceiling is an addition, not a modification. | fast |

J7 and J12 are differential: they pin a stated behaviour against another run of
this same code rather than against an outside answer. They are marked so that a
later reader does not mistake them for correctness oracles — J1, J2, J3, J5, J6,
J8 and J11 are the cases that could catch a wrong coefficient, and J4 pins
properties that hold by definition.

## 11. Dependencies

No new runtime dependency: `numpy` and the resampler already here.

`scikit-learn` is in the `dev` extra, as a **test oracle only** (D9). The rule it
serves is that a reference value must not come from the same code as the answer;
a hand-worked table alone is one witness, and a second independent
implementation is the other. It is imported by `tests/test_agreement.py` and
nowhere under `src/`, and the comment beside it in `pyproject.toml` says so, so
that a later reader does not promote it to a runtime dependency by accident. The
dependency-thrift rule is untouched.
