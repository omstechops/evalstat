"""Agreement between two raters on an ordered scale, resampled by cluster.

The question this answers is whether an LLM judge can stand in for a human
rater. Raw match rate cannot answer it: two raters who both call almost
everything "equal" match almost always, and that match rate says nothing about
the items that discriminate between systems. Cohen's kappa answers it by
subtracting the agreement two raters would reach independently at those same
base rates, which is what makes it a ratio of estimated quantities and gives it
the failure modes this module reports beside it rather than behind it.

Clustering is the reason it is written here rather than called from elsewhere.
``sklearn.metrics.cohen_kappa_score`` computes the coefficient and stops; there
is no interval, and no notion that three items cut from one recording carry a
shared component. An interval that resamples items is too narrow when that
shared component reaches the *agreement* -- when some recordings are hard for
both raters at once -- and the whole point of quoting an agreement figure is to
say how far it could be from the truth. When only the labels share a component
and the disagreement does not, the naive interval turns out to be nearly right;
the table below is the measurement of both cases, and it is not the same
picture :mod:`evalstat.bootstrap` paints for the mean.

Documented coverage finding
---------------------------
Empirical coverage of nominal 95% percentile intervals, measured by the
simulation in
``tests/test_agreement_interval.py::test_j11_coverage_depends_on_what_is_clustered``:
300 datasets at k = 40 clusters, m = 3 items each, linear-weighted kappa on
three categories, 499 resamples per interval. The latent quality behind the
ratings has intra-cluster correlation 0.5 at every row; what varies down the
table is ``difficulty_sd``, the spread of a cluster-level factor on both raters'
noise, reported beside the ICC it produces in the per-item disagreement
indicator. The population kappa of each row is ``sklearn``'s figure on 150,000
items from the same generator.

=============  =====  =====  ==================  =============  ===========
difficulty_sd  ICC    kappa  Item-level (naive)  Cluster-level  Width ratio
=============  =====  =====  ==================  =============  ===========
0.0            0.013  0.506  0.920               0.927          1.04
0.7            0.135  0.483  0.940               0.943          1.10
2.0            0.262  0.460  0.920               0.950          1.18
=============  =====  =====  ==================  =============  ===========

Monte Carlo standard error on each coverage figure is about 0.013; the width
ratio (mean clustered width over mean naive width) is known far more precisely,
since width is deterministic per dataset. Three things follow.

**Whether clustering matters to kappa depends on what is clustered.** The
first row has the labels clustered as strongly as the others -- a latent ICC of
0.5 -- and the two intervals are the same to within noise, 4% apart in width.
Kappa is a ratio of an observed to an expected disagreement, and a component
shared by every item of a cluster moves both and largely cancels. What the
correction needs is a between-cluster component in the *disagreement itself*:
at a disagreement ICC of 0.26 the clustered interval is 18% wider, the naive
one covers 0.92 and the clustered one 0.95. So ``cluster=`` is not a formality
here the way it is for a mean, and neither is it harmless to omit: the study's
recordings will differ in how contestable they are, and that is exactly the
component the first row lacks. Which row a given dataset sits in is a fact
about that dataset, and the result carries it as ``disagreement_icc`` so that
an interval quoted from this function can be placed in this table.

**Neither interval reaches nominal in the first row.** At 120 items and forty
clusters the percentile interval of a kappa covers about 0.92-0.93 with no
clustering to blame, which is the small-sample behaviour of a bootstrapped
ratio rather than anything about clusters. A study reporting a 95% interval
from this function at this size should say so.

**Only the percentile method is measured.** The paired bootstrap's table found
the three methods indistinguishable at forty clusters and chose the one with
the fewest estimated quantities; this module inherits that default rather than
re-measuring it, and the inheritance is open to revision if someone does.

Why this is not a call into ``paired_bootstrap``
------------------------------------------------
Because ``x - y`` destroys what the coefficient needs. On a five-point scale the
pairs (1, 2) and (4, 5) both give a difference of -1, and they contribute
differently to expected agreement: how much agreement is due to chance depends
on *which* categories the two raters used, not only on how far apart they were.
So the statistic here needs both label vectors, not their difference.

That is the contract :mod:`evalstat._resample` exists for. The resampler is
keyed on item indices, so this module's statistic closes over both vectors and
indexes each -- ``lambda idx: _kappa(a[idx], b[idx], weight)`` -- while the
drawing of whole clusters stays in one place, shared with the paired case. The
resampler is handed ``unit="rated item"`` with it, so a mis-shaped ``cluster``
argument is reported in this module's vocabulary rather than in the paired
case's. What is *not* shared is the argument contract, and that separation is
deliberate: :func:`evalstat.paired_bootstrap` rules out hosting a second
contract in its own docstring.

One function, two shapes of rating task
---------------------------------------
Preference judgements (A better / equal / B better) and absolute scores (1 to 5)
are the same mathematics on a different number of ordered categories, so they
are one function and not two. What makes that safe is that ``categories`` is
required and given in order: the category set comes from the rating design, not
from the data. A category nobody happened to use still exists in the design, and
inferring the set from the data drops it, shrinks the weight matrix and re-spaces
the scale. Whether that moves the coefficient depends on what is left: a label
missing from one end rescales every distance by the same factor and cancels out,
while a five-point scale used only at 1, 2 and 5 is re-spaced unevenly and the
coefficient moves. The table and the prevalence index change either way, and
nothing in the output says which case occurred.

Why the human ceiling is not optional
--------------------------------------
A judge-human kappa read on its own is uninterpretable, and it is not an upper
bound on anything. Kappa's chance model treats the two raters' marginals as
independent; an LLM judge trained on human preference data is not independent of
a human in that sense, and the bias the two share is not counted as chance by
that model, so it arrives as agreement. The coefficient is therefore biased
upward by an amount that cannot be estimated from the table it was computed on.

The response is a comparison, not a correction: ``ceiling=`` takes a second
human's labels on the same items, and the human-human kappa measured there is
what the judge's figure is read against. The difference between the two is
reported -- not their ratio, whose denominator is small and unstable -- and both
come from the same resamples, because they are computed on the same items and
are correlated. Two independently drawn intervals would give a difference
interval that is wrong in a direction nobody would check.

"The same resamples" is structural, not reproduced: the index sets are drawn
once and the three statistics -- judge kappa, human kappa, their difference --
are evaluated on that one set of draws. Re-seeding would reproduce the draws
today and would silently stop doing so for a caller-supplied generator, whose
stream simply continues; see :mod:`evalstat._resample`. A resample in which
either kappa is undefined is discarded from all three, so every number in the
result comes from one and the same set of valid resamples and ``n_valid``
counts that set. The judge's own interval can therefore differ slightly
between a call with ``ceiling=`` and one without, when a resample the human
pair could not define is dropped; that is the cost of not carrying two
different samplings in one result object, and it is paid knowingly.

Not here, deliberately
----------------------
Krippendorff's alpha, Gwet's AC1/AC2, PABAK, more than two raters, and any
reading of agreement as correctness. The reasons are in
``docs/design/judge_agreement.md``; the short version of the last one is
assumption 12.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evalstat._resample import (
    Method,
    check_resampling_arguments,
    cluster_bootstrap,
    draw_resamples,
    evaluate,
)

__all__ = [
    "HumanCeiling",
    "JudgeAgreementResult",
    "Weights",
    "judge_agreement",
]

Weights = Literal["linear", "unweighted"]
"""Disagreement weighting. Required at every call site; there is no default."""


@dataclass(frozen=True)
class HumanCeiling:
    """The human-human kappa, and the judge's distance from it.

    Both coefficients are computed on the same items and from the same
    resamples, so ``difference`` and its interval are paired quantities. They
    cannot be reconstructed from the two kappas' intervals, which is why the
    comparison is an argument to :func:`judge_agreement` rather than a second
    call to it.

    Attributes
    ----------
    kappa
        Agreement between the two human raters, on the same scale and weighting
        as the judge's.
    ci_low, ci_high
        Interval for ``kappa``.
    difference
        ``judge kappa - human kappa``. Negative means the judge agrees with the
        human less than a second human does. A ratio of the two is not reported:
        see the module docstring.
    difference_ci_low, difference_ci_high
        Interval for ``difference``, from the resamples both kappas were
        computed on. Narrower than an interval built as though the two were
        independent, because they are not.
    distribution
        The resampled human-human coefficients, aligned resample by resample
        with the judge's ``distribution`` on the parent result.
    difference_distribution
        The resampled differences, one per valid resample. Equal, element by
        element, to the judge's ``distribution`` minus ``distribution`` here;
        that equality is what "the same resamples" means, and it is what the
        tests check rather than the mechanism that produces it.
    """

    kappa: float
    ci_low: float
    ci_high: float
    difference: float
    difference_ci_low: float
    difference_ci_high: float
    distribution: NDArray[np.float64]
    difference_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class JudgeAgreementResult:
    """One agreement coefficient, its interval, and the table behind it.

    The diagnostics are not optional extras and are populated on every call. A
    kappa is a ratio, and the same value arises from a table that is well
    balanced and from one that is not; ``p_observed``, ``prevalence_index``,
    ``bias_index`` and ``table`` are what separate those cases. Reporting the
    coefficient alone leaves the reader to guess which one they are looking at.

    Attributes
    ----------
    kappa
        Weighted agreement coefficient. 1.0 is exact agreement on every item; 0
        is the agreement the chance model expects; negative values mean the two
        raters agree less than that model expects.
    ci_low, ci_high
        Interval endpoints, from resampling whole clusters.
    confidence_level
        Nominal coverage requested. See :mod:`evalstat.bootstrap` for what a
        nominal level is worth at a small number of clusters.
    method
        Interval construction used: ``percentile``, ``basic`` or ``bca``.
    weights
        Weighting the coefficient was computed under. Carried on the result
        because a weighted kappa and an unweighted one are different quantities
        and a number quoted without its scheme cannot be compared to anything.
    categories
        The ordered category set, as supplied.
    p_observed
        Weighted observed agreement, before any chance correction. The quantity
        a naive "how often do they match" report would show.
    p_expected
        Weighted agreement the chance model expects from the two raters'
        marginals. ``kappa`` is ``(p_observed - p_expected) / (1 -
        p_expected)``, so a large ``p_expected`` is what turns a high match rate
        into a low coefficient.
    prevalence_index
        How unevenly agreement sits across the scale, as ``max_i p_ii - min_i
        p_ii``. Large when the raters agree overwhelmingly in one category,
        which is the condition under which kappa falls while ``p_observed``
        stays high. At two categories this is the standard ``abs(a - d) / n``;
        past two it is **a** generalisation of that definition and not the only
        one, and the other generalisations give different numbers. It is also a
        summary of the diagonal and nothing more -- a declared category nobody
        used contributes a zero to the minimum -- so where the diagnosis
        matters, read ``table``. The index points; the table is the evidence,
        and it is returned unconditionally for that reason.
    bias_index
        How far the two raters' marginal distributions are from each other. A
        judge that is systematically harsher than the human scores a high bias
        index, and that is a different defect from disagreeing at random.
    disagreement_icc
        Intra-cluster correlation of the per-item weighted disagreement: how
        much of the variance in "how far apart were the two raters on this
        item" sits between clusters. One-way ANOVA ICC(1), on the unbalanced
        form so that clusters of unequal size are handled; it can be negative,
        and is reported as computed. ``nan`` when ``cluster`` is None -- there
        is no within-cluster variance to compare against -- and when the
        disagreement is constant across every item.

        This is the fourth diagnostic, not a general-purpose ICC: it is defined
        on the disagreement indicator under the ``weights`` in force and on
        nothing else, and it exists because the coverage table in the module
        docstring turns on it. A reader placing an interval from this
        function in that table needs this number beside it. The intraclass
        correlation of a continuous outcome is a different function, and it is
        not this one.
    table
        Cross-tabulation of counts, shape ``(n_categories, n_categories)``.
        **Rows are the judge, columns are the human**, in the order given by
        ``categories``. Off-diagonal mass above the diagonal therefore means the
        judge rated higher.
    n_items
        Items rated by both raters.
    n_clusters
        Number of resampling units. Equals ``n_items`` when ``cluster`` is None.
    cluster_sizes
        ``(minimum, median, maximum)`` items per cluster.
    n_resamples
        Resamples requested.
    n_valid
        Resamples that yielded a finite coefficient and entered the interval.
        With ``ceiling``, resamples on which *both* coefficients were finite:
        one set of valid resamples serves every number in the result.
    seed
        Seed the resampling ran from. Populated even when the caller passed
        nothing; ``None`` only when the caller supplied its own
        :class:`numpy.random.Generator`.
    distribution
        The ``n_valid`` resampled coefficients.
    ceiling
        The human-human comparison, or ``None`` when ``ceiling`` was not passed.
        A result with ``None`` here carries a number that assumption 6 says
        cannot be read on its own.
    """

    kappa: float
    ci_low: float
    ci_high: float
    confidence_level: float
    method: str
    weights: str
    categories: tuple[object, ...]
    p_observed: float
    p_expected: float
    prevalence_index: float
    bias_index: float
    disagreement_icc: float
    table: NDArray[np.int64]
    n_items: int
    n_clusters: int
    cluster_sizes: tuple[int, int, int]
    n_resamples: int
    n_valid: int
    seed: int | None
    distribution: NDArray[np.float64]
    ceiling: HumanCeiling | None


def _weight_matrix(n_categories: int, weights: Weights) -> NDArray[np.float64]:
    """Disagreement cost between every ordered pair of categories.

    ``linear`` charges ``abs(i - j) / (n_categories - 1)``: the cost of a
    disagreement is proportional to how many steps apart the two categories sit
    on the scale. ``unweighted`` charges 1 for any disagreement, which throws
    the ordering away and is the reason ``weights`` has no default.

    The normalisation is cosmetic. Kappa is a ratio of two sums against the same
    matrix, so scaling every cost by a constant leaves it unchanged; dividing by
    ``n_categories - 1`` only keeps the entries in ``[0, 1]`` where they are
    easier to read.
    """
    position = np.arange(n_categories)
    distance = np.abs(position[:, None] - position[None, :]).astype(np.float64)
    if weights == "linear":
        return distance / (n_categories - 1)
    return (distance > 0.0).astype(np.float64)


def _cross_table(
    a: NDArray[np.intp], b: NDArray[np.intp], n_categories: int
) -> NDArray[np.int64]:
    """Count each ordered pair of codes, rows ``a`` and columns ``b``.

    Every declared category keeps its row and column whether or not anyone used
    it, which is what makes the weight matrix line up with the table: a category
    dropped here would shift every position after it and quietly re-space the
    scale. See assumption 2.
    """
    counts = np.bincount(a * n_categories + b, minlength=n_categories**2)
    return counts.reshape(n_categories, n_categories).astype(np.int64)


def _encode(
    values: ArrayLike, lookup: dict[object, int], name: str
) -> NDArray[np.intp]:
    """Turn labels into positions in ``categories``.

    Positions, not labels, are what the arithmetic uses, and resolving them once
    outside the resampling loop is what stops the scale moving between
    resamples. A label the design did not declare is an error rather than a new
    category: inventing one here would silently widen the scale.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    codes = np.empty(array.size, dtype=np.intp)
    for i, label in enumerate(array.tolist()):
        try:
            codes[i] = lookup[label]
        except KeyError:
            raise ValueError(
                f"{name} contains {label!r}, which is not in categories "
                f"{list(lookup)!r}. The category set comes from the rating "
                "design, so a label outside it is a data error, not a category "
                "this function may add"
            ) from None
    return codes


def _disagreement_icc(
    disagreement: NDArray[np.float64], cluster: ArrayLike | None
) -> float:
    """One-way ICC(1) of a per-item quantity across clusters, unbalanced form.

    ``MS_between`` and ``MS_within`` are the usual one-way ANOVA mean squares;
    with unequal cluster sizes the "items per cluster" in the ICC formula is
    replaced by ``m0 = (N - sum(n_c**2) / N) / (k - 1)``, which reduces to
    ``m`` when every cluster has ``m`` items. Returns ``nan`` when there is
    nothing within clusters to compare (no clustering, or every cluster a
    single item) and when both mean squares are zero.

    Runs after the resampler has validated ``cluster``, so it does not repeat
    the shape check and would not be reached with a malformed one.
    """
    if cluster is None:
        return float("nan")
    _, codes = np.unique(np.asarray(cluster), return_inverse=True)
    codes = codes.ravel()
    n_items = disagreement.size
    sizes = np.bincount(codes).astype(np.float64)
    n_clusters = sizes.size
    if n_items == n_clusters or n_clusters < 2:
        return float("nan")
    means = np.bincount(codes, weights=disagreement) / sizes
    grand = float(disagreement.mean())
    ms_between = float((sizes * (means - grand) ** 2).sum()) / (n_clusters - 1)
    ms_within = float(((disagreement - means[codes]) ** 2).sum()) / (
        n_items - n_clusters
    )
    m0 = (n_items - float((sizes**2).sum()) / n_items) / (n_clusters - 1)
    denominator = ms_between + (m0 - 1.0) * ms_within
    if denominator == 0.0:
        return float("nan")
    return (ms_between - ms_within) / denominator


def _kappa(
    a: NDArray[np.intp],
    b: NDArray[np.intp],
    weight: NDArray[np.float64],
) -> float:
    """Weighted kappa for two vectors of category codes.

    This is the function the resampler is handed, closed over both vectors:
    ``lambda idx: _kappa(a[idx], b[idx], weight)``. It takes codes rather than
    labels so that the category set is resolved once, outside the resampling
    loop, and cannot shift between resamples.

    Returns ``nan`` when the expected disagreement is zero, which happens when a
    resample leaves both raters on one and the same category. That is the only
    way the denominator vanishes: ``sum(w * outer(r, c))`` is zero only if the
    weight is zero wherever the two marginals have mass together, and the weight
    is zero only on the diagonal. It is the shared machinery's contract for an
    undefined resample: discarded, counted, and turned into a warning or --
    past :data:`evalstat.MIN_VALID_FRACTION` -- an error. The *observed* table
    is the same arithmetic and a different decision, checked before any
    resampling begins; see :func:`judge_agreement`.
    """
    n_categories = int(weight.shape[0])
    observed = _cross_table(a, b, n_categories) / a.size
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    disagreement_expected = float((weight * expected).sum())
    if disagreement_expected == 0.0:
        return float("nan")
    return 1.0 - float((weight * observed).sum()) / disagreement_expected


def judge_agreement(
    judge: ArrayLike,
    human: ArrayLike,
    *,
    categories: Sequence[object],
    weights: Weights,
    cluster: ArrayLike | None = None,
    ceiling: ArrayLike | None = None,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    method: Method = "percentile",
    rng: int | np.random.Generator | None = None,
) -> JudgeAgreementResult:
    """Weighted kappa between a judge and a human, with a cluster interval.

    Both preference judgements and absolute scores are rated tasks over an
    ordered category set, and this is one function for both. What distinguishes
    them is ``categories``, which is required: see the module docstring for why
    it is not inferred from the data.

    Parameters
    ----------
    judge, human
        Labels for the same items, aligned by position, each drawn from
        ``categories``. The coefficient itself is symmetric in its two
        arguments; the names record the roles the study assigns, and the roles
        are what assumption 6 is about.
    categories
        The complete category set **in ascending order**, from the rating
        design. Order carries the scale: distances are counted in positions in
        this sequence, so a list given out of order silently computes a
        different coefficient. Labels not appearing in the data are kept.
    weights
        ``linear`` or ``unweighted``, required, no default. The only default
        available would be ``unweighted``, which discards the ordering and is
        exactly the error worth avoiding on an ordered scale -- the same reason
        ``rho`` has no default in :func:`evalstat.power_analysis`. ``quadratic``
        is not offered: it is forgiving of adjacent disagreement, and on a
        preference scale the confusion between "equal" and "A slightly better"
        is the thing being measured, not a rounding error.
    cluster
        One label per item. Items sharing a label are resampled together. Three
        rated items cut from one recording, or three questions from one article,
        are one cluster. ``None`` means every item is its own cluster.
    ceiling
        A second human's labels for the same items, in the same categories. When
        given, the human-human kappa is computed from the same resamples and the
        signed difference is reported on :class:`HumanCeiling`. Without it the
        result carries a coefficient that, by assumption 6, has no scale to be
        read against.
    n_resamples
        Cluster draws to take.
    confidence_level
        Nominal coverage.
    method
        ``percentile`` (default), ``basic`` or ``bca``, as in
        :func:`evalstat.paired_bootstrap`.
    rng
        Seed, generator, or ``None``. When ``None`` a seed is drawn and recorded
        on the result, so a run stays reproducible after the fact.

    Returns
    -------
    JudgeAgreementResult
        The coefficient, its interval, the diagnostics that say how to read it,
        and the ceiling comparison when one was requested.

    Raises
    ------
    ValueError
        If ``judge``, ``human``, ``ceiling`` or ``cluster`` do not all describe
        the same items; if a label appears in the data that is not in
        ``categories``, or ``categories`` repeats a label or holds fewer than
        two; if ``weights`` is not one of the two offered; if the resampling
        arguments are malformed or fewer than two clusters are present.

        If the **observed** table is degenerate -- both raters constant, or one
        rater constant, so that expected agreement is 1 and kappa's denominator
        vanishes -- the error names that state rather than returning ``nan``. A
        table like that is a fact about the data, not a numerical accident, and
        ``nan`` invites being read as a number. The same condition inside a
        *resample* is not an error; see :func:`_kappa`.

        If more than ``1 - MIN_VALID_FRACTION`` of resamples were undefined.

    Warns
    -----
    evalstat.FewClustersWarning
        Fewer clusters than :data:`evalstat.MIN_CLUSTERS`.
    evalstat.DegenerateResampleWarning
        Some, but not too many, resamples left a rater constant and were
        discarded.

    Notes
    -----
    Assumptions, and where each one fails:

    1. **The scale is ordinal and its steps are equally spaced.** Linear weights
       charge a disagreement in proportion to the number of steps between the
       two categories, which claims the gap from "equal" to "A slightly better"
       is the same size as the gap from "A slightly better" to "A much better".
       Anchors written to make that roughly true are a design decision, not a
       property of the numbers.
    2. **``categories`` is the design's category set, complete and in order.**
       Inferring it from the data re-spaces the scale, and whether that shows up
       in the coefficient depends on the data: a label missing from one end
       rescales every distance by one factor and cancels, while a five-point
       scale used only at 1, 2 and 5 is re-spaced unevenly and the coefficient
       moves. The table and the prevalence index change in both cases. Passing
       the labels out of order computes a coefficient for a scale nobody rated
       on.
    3. **Every item was rated by both raters.** Items one rater skipped are not
       a shorter array: dropping them conditions the coefficient on the items
       that were easy enough to rate, which are the items agreement is cheapest
       on.
    4. **Chance agreement is the product of the two raters' marginals.** This is
       Cohen's model, and it is a model. Under it a rater's own base rate is
       subtracted as chance, so a table with skewed marginals yields a low kappa
       at a high match rate -- the kappa paradox. That is why ``p_observed``,
       the two indices and the table are reported unconditionally rather than on
       request.
    5. **The marginals are estimated from the same table as the agreement.**
       Kappa is a ratio of estimated quantities, so it is biased in small
       samples and unstable when some category is rated a handful of times. The
       interval covers the sampling variation, not this bias.
    6. **The judge and the human are not independent, so a judge-human kappa is
       not an upper bound and not an unbiased estimate: it is biased upward by
       an unknown amount.** The chance model in assumption 4 counts as chance
       only what two raters would match on by hitting the same category at their
       own base rates. An LLM judge trained on human preference data shares the
       human's biases -- the same items look hard to both, the same phrasing
       flatters both -- and shared bias is not chance under that model, so every
       item it decides arrives as agreement. Nothing in the table separates the
       shared-bias part from the genuine-agreement part, and no amount of data
       shrinks it, because it is bias and not variance.

       **This is what makes the human ceiling comparison mandatory rather than a
       nicety.** A judge kappa of 0.71 is not a fact about the judge until it is
       read against what two humans reach on the same items under the same
       scale: at a human-human 0.74 the judge is close to the ceiling of the
       task, and at a human-human 0.90 it is not close to anything. The absolute
       value cannot make that distinction, and calling it a "lower bound on
       human-level agreement" gets the direction backwards. Hence ``ceiling=``,
       hence the difference reported from the same resamples, and hence the
       requirement that the human labels be produced independently of the judge:
       a human who has seen the judge's label is no longer a second measurement
       of the task.
    7. **Clusters are mutually independent, and dependence within a cluster may
       take any form.** Two recordings of one speaker treated as separate
       clusters break this, as does one rater who worked through the items in
       cluster order and drifted.
    8. **Consistency is in the number of clusters, not the number of items.**
       Rating more items inside the same clusters buys precision the interval is
       not entitled to. See :data:`evalstat.MIN_CLUSTERS`.
    9. **Kappa is a ratio and can be undefined, in exactly one situation.** The
       denominator ``1 - p_expected`` vanishes only when *both* raters used one
       and the same category throughout: the chance model then expects perfect
       agreement, and the coefficient is 0/0. One constant rater is not that
       case -- it gives exactly 0, which is the right answer, since a rater who
       always says the same thing agrees exactly as often as chance predicts.
       Observed: an error naming the state. Resampled: discarded, counted, and
       an error if too many.
    10. **This returns an interval, not a test.** It carries no null hypothesis,
        no multiplicity correction across the several rubric dimensions a study
        usually has, and no threshold. The verbal labels attached to kappa
        ranges in the literature ("substantial", "almost perfect") are
        convention with no sampling justification, and a study that needs a
        pass mark has to set and pre-register one.
    11. **A weighted kappa is not comparable to an unweighted one**, nor to one
        computed on a different category set. The scheme and the categories are
        carried on the result so a reported number keeps them.
    12. **This measures agreement, not correctness.** Two raters can agree and
        both be wrong, and against a fixed human label set the highest-scoring
        judge is the one that reproduces the humans' errors most faithfully.
        Agreement is evidence that the judge can stand in for the human, which
        is a claim about substitutability and not about truth.

    The decisions behind this interface, the alternatives rejected and the
    questions still open are in ``docs/design/judge_agreement.md``.
    """
    # Ahead of the data, so that a malformed n_resamples is reported as such
    # whether or not the data is also malformed; cluster_bootstrap checks the
    # same three again, from the same definition.
    check_resampling_arguments(n_resamples, confidence_level, method)

    if weights not in ("linear", "unweighted"):
        # Widened deliberately: the annotation says this cannot happen and the
        # runtime is where it does, so the comparison below has to survive the
        # type checker narrowing `weights` to nothing.
        supplied: object = weights
        offered = "'linear' or 'unweighted'"
        if supplied == "quadratic":
            raise ValueError(
                f"weights must be {offered}; quadratic weighting is not offered "
                "at all. It charges a neighbouring disagreement a quarter of "
                "what it charges a two-step one, and on a preference scale the "
                "confusion between adjacent categories is the thing being "
                "measured. See docs/design/judge_agreement.md"
            )
        raise ValueError(f"weights must be {offered}, got {supplied!r}")

    declared = list(categories)
    if len(declared) < 2:
        raise ValueError(
            f"categories needs at least two labels, got {len(declared)}. A "
            "scale with one category records no distinction, so there is no "
            "agreement to measure"
        )
    lookup: dict[object, int] = {}
    for position, label in enumerate(declared):
        if label in lookup:
            raise ValueError(
                f"categories repeats {label!r}. Distances are counted in "
                "positions on the scale, so a repeated label would occupy two "
                "positions and sit at a distance from itself"
            )
        lookup[label] = position

    judge_codes = _encode(judge, lookup, "judge")
    human_codes = _encode(human, lookup, "human")
    if human_codes.shape != judge_codes.shape:
        raise ValueError(
            f"judge and human must rate the same items; got "
            f"{judge_codes.shape} and {human_codes.shape}. Rating must be "
            "complete: an item one rater skipped breaks the design, it does "
            "not shorten it"
        )
    n_items = int(judge_codes.size)
    if n_items == 0:
        raise ValueError("need at least one rated item")

    if ceiling is not None:
        ceiling_codes = _encode(ceiling, lookup, "ceiling")
        if ceiling_codes.shape != judge_codes.shape:
            raise ValueError(
                f"ceiling must rate the same items as judge and human; got "
                f"{ceiling_codes.shape} and {judge_codes.shape}. The ceiling is "
                "measured on the items the judge was measured on, or it is not "
                "the ceiling of this measurement"
            )

    n_categories = len(declared)
    weight = _weight_matrix(n_categories, weights)
    table = _cross_table(judge_codes, human_codes, n_categories)
    observed = table / n_items
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    disagreement_observed = float((weight * observed).sum())
    disagreement_expected = float((weight * expected).sum())

    if disagreement_expected == 0.0:
        # The only way the denominator vanishes: see _kappa. Naming the state is
        # the whole point of raising here rather than returning nan -- the
        # caller has a rubric dimension nobody discriminated on, and needs to be
        # told that rather than handed a number-shaped absence.
        used = declared[int(np.argmax(np.diag(table)))]
        raise ValueError(
            f"both raters used the category {used!r} for all {n_items} rated "
            "items, so the chance model expects perfect agreement "
            "(p_expected = 1) and kappa is undefined: its denominator is zero. "
            "This is a fact about the data, not a numerical accident, so it is "
            "an error and not a NaN. One constant rater is a different case and "
            "is not an error: it gives exactly 0"
        )

    def judge_kappa(idx: NDArray[np.intp]) -> float:
        return _kappa(judge_codes[idx], human_codes[idx], weight)

    if ceiling is None:
        drawn = cluster_bootstrap(
            judge_kappa,
            n_items,
            cluster=cluster,
            unit="rated item",
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            rng=rng,
        )
        comparison = None
    else:
        ceiling_table = _cross_table(ceiling_codes, human_codes, n_categories)
        ceiling_observed = ceiling_table / n_items
        if (
            float(
                (
                    weight
                    * np.outer(
                        ceiling_observed.sum(axis=1), ceiling_observed.sum(axis=0)
                    )
                ).sum()
            )
            == 0.0
        ):
            used = declared[int(np.argmax(np.diag(ceiling_table)))]
            raise ValueError(
                f"both humans used the category {used!r} for all {n_items} "
                "rated items, so the human-human kappa is undefined and there "
                "is no ceiling to compare against; see the judge-human case "
                "above for why this is an error and not a NaN"
            )

        def human_kappa(idx: NDArray[np.intp]) -> float:
            return _kappa(ceiling_codes[idx], human_codes[idx], weight)

        def both(idx: NDArray[np.intp]) -> tuple[float, float]:
            # One validity rule for all three statistics: a resample on which
            # either coefficient is undefined is undefined for the judge, the
            # human and the difference alike, so the three passes below
            # discard the same resamples and n_valid counts one common set.
            # _kappa returns nan and never inf, so the difference carries the
            # rule on its own; the judge and human need it imposed.
            j, h = judge_kappa(idx), human_kappa(idx)
            if not (np.isfinite(j) and np.isfinite(h)):
                return float("nan"), float("nan")
            return j, h

        # Three scalar passes over one set of draws. Each pass computes both
        # kappas and keeps one, so a resample costs six kappa evaluations
        # instead of two. A kappa is one bincount over the resample and the
        # whole run is milliseconds; the alternative -- a statistic returning
        # three numbers -- would put a second contract into the resampler,
        # which is what the index-keyed core exists to avoid.
        draws = draw_resamples(
            n_items,
            cluster=cluster,
            unit="rated item",
            n_resamples=n_resamples,
            rng=rng,
        )
        drawn = evaluate(
            lambda idx: both(idx)[0],
            draws,
            confidence_level=confidence_level,
            method=method,
        )
        humans = evaluate(
            lambda idx: both(idx)[1],
            draws,
            confidence_level=confidence_level,
            method=method,
        )
        gap = evaluate(
            lambda idx: both(idx)[0] - both(idx)[1],
            draws,
            confidence_level=confidence_level,
            method=method,
        )
        comparison = HumanCeiling(
            kappa=humans.estimate,
            ci_low=humans.ci_low,
            ci_high=humans.ci_high,
            difference=gap.estimate,
            difference_ci_low=gap.ci_low,
            difference_ci_high=gap.ci_high,
            distribution=humans.distribution,
            difference_distribution=gap.distribution,
        )

    return JudgeAgreementResult(
        kappa=drawn.estimate,
        ci_low=drawn.ci_low,
        ci_high=drawn.ci_high,
        confidence_level=confidence_level,
        method=method,
        weights=weights,
        categories=tuple(declared),
        p_observed=1.0 - disagreement_observed,
        p_expected=1.0 - disagreement_expected,
        prevalence_index=float(np.diag(observed).max() - np.diag(observed).min()),
        bias_index=float(
            np.abs(observed.sum(axis=1) - observed.sum(axis=0)).sum() / 2.0
        ),
        disagreement_icc=_disagreement_icc(weight[judge_codes, human_codes], cluster),
        table=table,
        n_items=n_items,
        n_clusters=drawn.n_clusters,
        cluster_sizes=drawn.cluster_sizes,
        n_resamples=n_resamples,
        n_valid=drawn.n_valid,
        seed=drawn.seed,
        distribution=drawn.distribution,
        ceiling=comparison,
    )
