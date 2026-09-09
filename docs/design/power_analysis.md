# Design note — `power_analysis()`

**Status:** implemented, both routes. The analytic route landed first, the
simulation route after it, and the tests were written before either. Three gaps
in the simulation route are deliberate and recorded in section 4.

This note is the decision record. It says what was decided and why, and what is
still open. It does **not** repeat the assumption list: that lives in the
`power_analysis` docstring, where it is read by whoever calls the function, and
a second copy here would be a second copy to keep in step.

---

## 1. What it answers

Two directions of one relationship:

1. **Forwards.** How many clusters are needed to detect an effect of a stated
   size with probability `power`?
2. **Inverted.** With the design fixed by collection and rating capacity, what
   is the smallest effect it can detect?

The second is the question an evaluation usually faces, because capacity fixes
n long before anyone computes power. It is served by the same function through
the `statsmodels` idiom: exactly one of `effect`, `n_clusters`,
`items_per_cluster` and `power` is passed as `None`, and that is the one solved
for. The error message for a wrong count of `None`s names all four slots.

**MDE means the power-based quantity**: the effect detectable with probability
`power`. It is not the expected half-width of the interval. An effect equal to
the half-width is detected roughly half the time, so reporting a half-width as
an MDE describes a design at about 50% power as though it were at 80%. The
docstring rejects the alternative reading explicitly rather than merely not
implementing it.

## 2. Scope

**In:** the mean paired difference, and the preference rate among non-tied
items.

These are not one formula with two arguments, and the design keeps them apart
for three reasons:

1. The variance of a rate moves with the rate; the variance of a mean does not.
2. Ties make the rate's denominator random, and ties are themselves clustered.
3. The intra-cluster correlation of a preference decision and that of a
   continuous difference are different quantities measured on the same data.

**Out, deliberately:**

- **Equivalence (TOST) power.** A different calculation; deferred until both
  arms of the study that needs it exist.
- **Estimating rho from data** (`intraclass_correlation()`). This function
  *takes* rho. Needed before pilot analysis, not before this.
- **Multiplicity.** Pass an already-corrected `alpha`; nothing here builds one.
- **Small-sample cluster corrections** (wild cluster bootstrap). Absent from
  `paired_bootstrap` too, and stated in both places.

## 3. Clustering, and the ceiling

```
DE    = 1 + (m_bar - 1) * rho
n_eff = k * m_bar / DE          ->  k / rho  as m_bar grows
```

`m_bar` is the variance-weighted mean cluster size `sum(m**2) / sum(m)`, not the
plain mean. They coincide only for equal-sized clusters; elsewhere the plain
mean is smaller and understates the design effect, which is the direction that
flatters the design.

`design_effect()` and `effective_n()` are public so that effective sample sizes
quoted in a plan are computed rather than typed.

The ceiling is given teeth: when solving for `items_per_cluster` and the target
exceeds `k / rho`, the function raises and names the ceiling. Returning a very
large item count would answer an impossible request with a number.

## 4. Analytic or simulation

| | Analytic | Simulation |
|---|---|---|
| What it does | normal approximation on `n_eff` | generates data, runs `paired_bootstrap`, counts rejections |
| Cost | microseconds | seconds to minutes |
| Mean | sound | confirms the analytic answer |
| Preference rate | fragile (section 2) | the right tool |
| Coverage shortfall | must be corrected by hand | inside the measurement |

**Default is `simulation`**, for both statistics. The analytic route stays
available, is the number other tools report, and is worth a comparison; it is
not the route to report from.

Simulation is affordable for the mean because the mean is equivariant under a
location shift: adding `delta` to every difference shifts the bootstrap interval
by `delta`, so `0 not in [lo + delta, hi + delta]` is `-delta not in [lo, hi]`.
One bootstrap run per dataset yields the exact power curve over the whole effect
grid, smooth and monotone, which is what makes root-finding on it stable. The
zero point of that curve is the procedure's empirical error rate at that design,
so coverage is measured for free and reported on the result.

A preference rate is not location-equivariant and its grid is evaluated point by
point.

### Known gaps in the simulation route

Three calls the analytic route answers raise `NotImplementedError` on the
simulation route. All three are deliberate, none is needed by the study this
package was built for, and each error message names the analytic route. They are
written down here so that a later reader can tell a decision from an oversight.

| | Gap | Why it is open | What closing it needs |
|---|---|---|---|
| G1 | one-sided `alternative` | `paired_bootstrap` returns a two-sided interval. Turning it into a one-sided test at `alpha` means deciding which interval to build it from — `confidence_level = 1 - 2*alpha` is the standard correspondence, but that is a choice about the test, and D2 eliminated one-sided testing for this study anyway. | A decision on the correspondence, and a test pinning it. Two lines of code after that. |
| G2 | solving `n_clusters` | Each candidate design needs its own simulation, so a search costs one full run per step instead of one in total. The mean's location-equivariance does not help: it moves the effect, not the sample size. | A decision on the search budget, and whether an unconverged search reports or raises. |
| G3 | solving `items_per_cluster` | Same as G2. | Same as G2. |

The intended workflow while they are open is the one the error messages state:
size the design on the analytic route, then check the design it returns with a
simulation run. That is also the better order — the analytic answer is the
cheap one and the simulation is the one that measures the procedure.

**These are route gaps, not scope exclusions.** Section 2's "out, deliberately"
list is about things this function will not do at all; these three are things it
does on one route and not yet on the other.

## 5. The coverage shortfall

`paired_bootstrap`'s measurement: at k = 40, m = 3, rho = 0.2, a nominal 95%
cluster bootstrap interval covers about 92.7%, Monte Carlo standard error about
1.5 points.

**The direction, because it is easy to get backwards.** Under-coverage means the
interval is *narrow*. A narrow interval excludes zero more often, so the
procedure rejects more often than nominal — under the null (about 7% rather than
5%) and under the alternative alike. So a nominal calculation is not optimistic
about the rejection rate; if anything it is pessimistic. The optimism is in the
**claimed error rate**: holding the real error at 5% would require a wider
interval, which costs power at fixed n and enlarges the MDE. That is the sense
in which an MDE computed against the nominal level is optimistic.

Four routes were considered:

| | Route | Claim it supports | Cost | Weakness |
|---|---|---|---|---|
| A | nominal level, shortfall documented | "computed at the nominal alpha" | none | states a threshold for a test that is not the one being run, and puts the gap in a limitations paragraph |
| B | calibrated alpha from the measured coverage | "real error held at 5%" | none | calibrated at exactly one design point and carried to every other |
| C | simulate the actual procedure | "this procedure rejects at this rate in this design" | seconds to minutes | assumes the data-generating model |
| D | report A and C as a band | both | as C | gives no single number |

**Decision: C is primary, A is available as the comparison.**

A is the pattern this project exists to criticise: report against the nominal
claim and footnote the discrepancy. B is extrapolation from one measured point
and its validity across the n grid is unverified — and running the measurement
across the grid to fix that is C. C's cost, the objection that used to carry
weight, is largely removed by the equivariance argument in section 4.

The choice is not hard-coded. `coverage=` keeps the handle outside the function:
supplied, the analytic route computes at `1 - coverage` instead of at `alpha`.
It is rejected on the simulation route, which measures that same quantity itself,
where a supplied figure would be a second and contradictory claim about it. Which
route produced the numbers is carried on the result as `coverage_source`
(`nominal` / `supplied` / `measured`).

### Settled: which error rate the MDE is quoted at

C reports the procedure's power **at the error rate the procedure actually runs
at**, and reports that error rate beside it. It does not calibrate the interval's
level so that the real error rate equals `alpha` and then read the MDE off the
calibrated procedure. Both are defensible and they answer different questions;
under simulation the second is reachable without extrapolation, unlike B, since
it would be calibrated at the same design point.

**The power reported is the power of the procedure that will be run, not of a
hypothetically calibrated one.** A pre-registration's reporting rests on the
procedure the study will actually execute; the calibrated figure answers "what
if the real error rate were 5%", which is a question about a procedure nobody
will run. Nothing is hidden by this choice: `coverage_measured` puts the real
error rate on the result, so a reader who wants the calibrated reading can
construct it.

## 6. Decisions taken

| | Decision | Rationale |
|---|---|---|
| D1 | `power` defaults to 0.80 | Convention, not a finding. Its role is as the level at which an MDE curve is read; the deliverable is the curve, and 0.80 is the marked point on it. |
| D2 | `alternative` defaults to `two-sided` | A one-sided test declares the opposite direction unmeasurable. Where blinding is set up against the study owner's expectation, burying that expectation in the test would undo it. The estimand is a system comparison, not a directional bet. |
| D3 | `rho` required, scalar or sequence, no default | The only possible default is zero, which is the error this package exists to correct. `None` meaning "grid" was rejected: it makes the return shape depend on the input. |
| D4 | Simulation primary, analytic as comparison | Section 5. |
| D5 | MDE is power-based; the half-width reading is rejected in the docstring | The two are confused often enough that not implementing the wrong one is insufficient. |
| D6 | `variance_under` defaults to `null` (p = 0.5) | A pre-registration should be conservative about its own power. |
| D7 | Analytic route open for the preference rate, with a warning | It is the number other tools report, so a comparison is worth having. The warning names the direction of the error — variance too small, power too high, MDE too small — in the style of `FewClustersWarning`. |
| D8 | TOST power and ICC estimation deferred | Section 2. |
| — | Result arrays always have shape `(n_rho,)` | Same reason as D3: one shape to handle, whether `rho` was a scalar or a sequence. |
| — | A solved `n_clusters` is returned unrounded | Rounding decides which side of the target to land on. The caller makes that decision visibly. |

## 7. Convention: a warning names the direction of the error

Every warning this package raises says which way the error runs, not merely that
one is possible. `FewClustersWarning` says the interval is narrower than the
truth rather than wider; `AnalyticProportionWarning` says the variance is too
small, so the power is too high and the MDE too small. Two is early enough to
write the rule down instead of discovering it as a pattern later.

The reason is that a warning without a direction cannot be acted on. "This may
be inaccurate" leaves the reader to guess whether the number flatters the design
or damns it, and the guess usually flatters. A warning that names the direction
converts into a decision: a result already too close to a threshold in the
warned direction is not usable, and one wrong-side-of-safe can be kept.

## 8. Test plan

Every case pins the function against an answer known independently of it.

| # | Case | Where the known answer comes from | Speed |
|---|---|---|---|
| R1 | Convergence to the classical formula when there is no clustering | `rho = 0`, two-sided `alpha = 0.05`, power 0.80, `effect/sd = 0.5` gives `n = (z_0.975 + z_0.8)**2 / 0.25 = 31.40`, so 32. The z-values are written literally in the test. | fast |
| R1b | The documented gap against t-based tools | The same scenario in G*Power or `statsmodels` gives 34; the difference is the t correction. The test fixes the gap as a claim so that a user comparing numbers is not left guessing. | fast |
| R2 | `rho = 0` leaves the design effect at exactly 1.0 | The formula, and equality — not approximate agreement — between the clustered and unclustered paths. | fast |
| R3 | Ceiling and diminishing returns | k = 40, rho = 0.2: m = 3 gives 85.7; rho = 0.1 gives 100; m very large approaches 200; `n_eff < k / rho` for every m. | fast |
| R4 | The two directions invert each other | `solve n from effect`, then `solve effect from that n`, returns the starting effect within rounding. | fast |
| R5 | Scale invariance | Power is unchanged under `(effect, sd) -> (c*effect, c*sd)`; MDE scales linearly in `sd`. | fast |
| R6 | Preference rate against the exact binomial | At `rho = 0`, `tie_rate = 0` the clustered sign test reduces to the exact binomial, whose power comes from `scipy.stats.binom`, not from this code. | slow |
| R7 | Consistency with `paired_bootstrap` | At the solved MDE, generate clustered data, run `paired_bootstrap`, and count intervals excluding zero. Target: within 0.03 of the requested power. | slow |
| R8 | Impossible target | Solving for `items_per_cluster` above `k / rho` raises, and the message contains the ceiling's value. | fast |
| R9 | Input rejection | `rho` outside `[0, 1)`, power or alpha outside `(0, 1)`, `effect = 0`, `tie_rate = 1`, `sd` missing or supplied wrongly, `coverage` on the simulation route, zero or two `None` slots. | fast |
| R10 | Monotonicity | Power rises in n and in `abs(effect)`, falls in `rho`; MDE falls in n, rises in `tie_rate`. | fast |
| R11 | Ties cost information | `tie_rate = 0.4` reduces the informative sample by the `(1 - tie_rate)` factor applied to the already-clustered effective n, not to the nominal item count. | fast |

**R7 is a correctness test, not a decision rule.** It exists to catch an
implementation error: if the MDE returned is right, the empirical rejection rate
at that effect is the requested power. It was at one point considered as a way
of deciding between routes A and C by measurement; that is no longer its job,
because C was chosen directly on the argument in section 5. A future reader
should not mistake a threshold in a test file for an open decision.

## 9. Dependencies

No new ones. `numpy`, `scipy.stats.norm` (already used), `scipy.optimize.brentq`
for root-finding. The simulation route calls `paired_bootstrap`. The signature
borrows the `statsmodels` solve-for-`None` idiom; `statsmodels` itself is not a
dependency.
