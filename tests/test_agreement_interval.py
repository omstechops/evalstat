"""Tests for :func:`evalstat.agreement.judge_agreement`: the clustered interval.

Stage two of the suite. ``test_agreement.py`` pins the coefficient's arithmetic,
which is deterministic; this file pins what the shared cluster resampler does
with it, which is stochastic and slow. Splitting them means a red test here
cannot be a wrong coefficient, and a red test there cannot be a wrong interval.

J11 is the measurement this module exists for, in the shape
``test_bootstrap.py`` already uses: ratings generated with a cluster-level
shared component, coverage of nominal 95% intervals counted with and without
``cluster=``, against a population value computed by ``sklearn`` on a very large
draw from the same generator and never by this code.

It is run at three settings of one generator rather than one, because the first
body to land showed that the paired bootstrap's premise does not carry over to
kappa unchanged. Clustering the *labels* -- items of one recording sharing a
latent quality -- leaves the naive interval essentially correct, since kappa is a
ratio and the shared component cancels between its numerator and denominator.
What has to be clustered for ``cluster=`` to matter is the *agreement*: some
recordings have to be hard for both raters at once. The generator carries a
``difficulty_sd`` knob for exactly that, and the test measures both regimes so
that neither is mistaken for a broken resampler.

**The tolerances in J11 are provisional.** They are predictions about a body
that does not exist yet, and the rule this project works to is that a tolerance
is measured before it is defended: first check whether the estimator is biased,
then measure the spread at the setting the test fixes. When the body lands, the
figures go into the ``evalstat.agreement`` module docstring the way the paired
bootstrap's coverage table did, and these bounds are set from that measurement.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score

from evalstat import DegenerateResampleWarning, FewClustersWarning
from evalstat.agreement import judge_agreement

CATEGORIES = (0, 1, 2)
CUTS = (-0.7, 0.7)
RHO = 0.5
RATER_NOISE = 0.6


def clustered_ratings(
    rng: np.random.Generator,
    n_clusters: int,
    items_per_cluster: int,
    n_raters: int = 2,
    difficulty_sd: float = 0.0,
) -> tuple[list[NDArray[np.intp]], NDArray[np.intp]]:
    """Draw ordinal ratings that share a cluster-level component.

    Each item carries a latent quality ``a_i + e_ij`` with ``var(a) = RHO`` and
    ``var(e) = 1 - RHO``, so the latent variance is 1 whatever ``RHO`` is. Every
    rater sees that latent value through independent noise and cuts it into
    categories at ``CUTS``.

    Two things follow, and the test uses both. Raters agree because they are
    looking at the same latent value, not because they were built to agree; and
    the items of one cluster are alike, so an interval that resamples items
    rather than clusters counts information that is not there -- *for a
    statistic that depends on it*, and kappa turns out largely not to.

    ``difficulty_sd`` is what makes agreement itself clustered. Each cluster
    draws one difficulty ``z_c``, and every rater's noise on that cluster has
    standard deviation ``RATER_NOISE * exp(difficulty_sd * z_c)``: a hard
    recording is hard for everyone who rates it. At ``difficulty_sd = 0`` the
    factor is 1 and the two raters' noise is independent across clusters, which
    is the original generator; the disagreement rate is then the same in every
    cluster up to sampling, and clustering has nothing to act on.

    The joint distribution of any two raters' labels does not depend on ``RHO``
    -- the cluster component is part of the latent variance either way -- but it
    does depend on ``difficulty_sd``, so the population coefficient is estimated
    per setting.
    """
    latent = rng.normal(0.0, np.sqrt(RHO), size=(n_clusters, 1)) + rng.normal(
        0.0, np.sqrt(1.0 - RHO), size=(n_clusters, items_per_cluster)
    )
    noise_sd = RATER_NOISE * np.exp(
        difficulty_sd * rng.normal(0.0, 1.0, size=(n_clusters, 1))
    )
    raters = [
        np.digitize(
            (latent + rng.normal(0.0, 1.0, size=latent.shape) * noise_sd).ravel(),
            CUTS,
        ).astype(np.intp)
        for _ in range(n_raters)
    ]
    cluster = np.repeat(np.arange(n_clusters), items_per_cluster).astype(np.intp)
    return raters, cluster


def disagreement_icc(
    a: NDArray[np.intp], b: NDArray[np.intp], items_per_cluster: int
) -> float:
    """One-way ICC(1) of the linear disagreement indicator, by cluster.

    The quantity the test reports beside ``difficulty_sd``: how much of the
    variance in ``abs(a - b)`` sits between clusters. This is a description of
    the generator, computed on the same large draw as the population kappa, and
    it is what "agreement is clustered" means in a number. Items are assumed to
    lie in cluster-contiguous order, which is how ``clustered_ratings`` lays
    them out.
    """
    d = (np.abs(a - b) / (len(CATEGORIES) - 1)).reshape(-1, items_per_cluster)
    k, m = d.shape
    grand = d.mean()
    ms_between = m * float(((d.mean(axis=1) - grand) ** 2).sum()) / (k - 1)
    ms_within = float(((d - d.mean(axis=1, keepdims=True)) ** 2).sum()) / (k * (m - 1))
    return (ms_between - ms_within) / (ms_between + (m - 1) * ms_within)


def population(difficulty_sd: float) -> tuple[float, float]:
    """Estimate the generator's coefficient and disagreement ICC at one setting.

    50_000 clusters of 3 from the same generator, scored by ``sklearn``. The
    Monte Carlo error on this is far below the width of any interval J11
    tests, and the figure never passes through the code under test.

    As in ``test_agreement.py``, the weighting crosses as the string
    ``"linear"`` and the weight matrix is ``sklearn``'s own. A matrix built here
    would make the target of the coverage measurement depend on the code the
    measurement is about.
    """
    (a, b), _ = clustered_ratings(
        np.random.default_rng(90210), 50_000, 3, difficulty_sd=difficulty_sd
    )
    kappa = float(cohen_kappa_score(a, b, labels=list(CATEGORIES), weights="linear"))
    return kappa, disagreement_icc(a, b, 3)


def agree(judge: NDArray[Any], human: NDArray[Any], **kwargs: Any) -> Any:
    """Call with this file's fixed scale and weighting."""
    return judge_agreement(
        judge, human, categories=CATEGORIES, weights="linear", **kwargs
    )


# ---------------------------------------------------------------------------
# J11  clustering restores coverage, and the run records itself
# ---------------------------------------------------------------------------

N_DATASETS = 300
SIM_RESAMPLES = 499
DIFFICULTY_LEVELS = (0.0, 0.7, 2.0)


@pytest.mark.slow
def test_j11_coverage_depends_on_what_is_clustered() -> None:
    """Clustered labels leave the naive interval valid; clustered agreement does not.

    The premise of the module, measured at three settings of the same generator
    so that both halves of the finding are pinned. At ``difficulty_sd = 0`` the
    labels are clustered but the disagreement is not, and the two intervals have
    to come out the same: same coverage within Monte Carlo error, same width
    within a few percent. That half exists so that a later reader who sees "no
    effect of clustering" does not conclude the resampler is broken. As
    ``difficulty_sd`` rises the disagreement ICC rises with it, the clustered
    interval has to widen relative to the naive one, and at the top setting the
    naive interval has to under-cover where the clustered one does not.

    The precise assertions are on mean width. Width is deterministic per
    dataset and its mean over 300 is known to a fraction of a percent, whereas
    coverage at 300 datasets carries a standard error of about 0.013, so the
    coverage bands are loose and the direction is carried by width. Neither
    interval reaches nominal at forty clusters and 120 items, at any setting;
    the table in the ``evalstat.agreement`` docstring records what they reach.

    The table this prints is that docstring's source, in the shape the paired
    bootstrap's table has.
    """
    rows = []
    for difficulty_sd in DIFFICULTY_LEVELS:
        truth, icc = population(difficulty_sd)
        naive_covered = clustered_covered = 0
        naive_width = clustered_width = 0.0
        for seed in range(N_DATASETS):
            (judge, human), cluster = clustered_ratings(
                np.random.default_rng(7000 + seed), 40, 3, difficulty_sd=difficulty_sd
            )
            naive = agree(judge, human, n_resamples=SIM_RESAMPLES, rng=8000 + seed)
            clustered = agree(
                judge,
                human,
                cluster=cluster,
                n_resamples=SIM_RESAMPLES,
                rng=8000 + seed,
            )
            naive_covered += naive.ci_low <= truth <= naive.ci_high
            clustered_covered += clustered.ci_low <= truth <= clustered.ci_high
            naive_width += naive.ci_high - naive.ci_low
            clustered_width += clustered.ci_high - clustered.ci_low
        rows.append(
            (
                difficulty_sd,
                icc,
                truth,
                naive_covered / N_DATASETS,
                clustered_covered / N_DATASETS,
                naive_width / N_DATASETS,
                clustered_width / N_DATASETS,
            )
        )

    print(
        "\ndifficulty_sd  ICC    kappa  naive  clustered  width_naive  width_clustered"
    )
    for sd, icc, truth, nc, cc, nw, cw in rows:
        print(
            f"{sd:13.1f}  {icc:.3f}  {truth:.3f}  {nc:.3f}  {cc:.3f}"
            f"      {nw:.3f}        {cw:.3f}"
        )

    iccs = [row[1] for row in rows]
    width_ratios = [row[6] / row[5] for row in rows]
    assert iccs == sorted(iccs), "difficulty_sd did not order the disagreement ICC"
    assert width_ratios == sorted(width_ratios), (
        "the clustered interval did not widen with the disagreement ICC"
    )

    _, icc_0, _, naive_0, clustered_0, _, _ = rows[0]
    assert icc_0 < 0.03, f"disagreement ICC {icc_0:.3f} at difficulty_sd = 0"
    assert width_ratios[0] < 1.06, (
        f"width ratio {width_ratios[0]:.3f} at difficulty_sd = 0: clustering "
        "acted on something, and there was nothing for it to act on"
    )
    assert abs(clustered_0 - naive_0) < 0.04

    _, icc_top, _, naive_top, clustered_top, _, _ = rows[-1]
    assert icc_top > 0.20, f"disagreement ICC {icc_top:.3f} at the top setting"
    assert width_ratios[-1] > 1.12
    assert naive_top < 0.935, (
        f"naive coverage {naive_top:.3f} did not under-cover with agreement "
        "clustered, which is the whole premise of resampling clusters"
    )
    assert clustered_top > naive_top
    assert 0.92 <= clustered_top <= 0.98


def test_j11_clustered_interval_is_wider_than_the_naive_one() -> None:
    """The fast half of the same claim, on one dataset rather than three hundred.

    Under-coverage and narrowness are the same fact seen from two sides, and
    this direction is cheap to check: the naive interval, counting 120 items as
    120 independent observations, has to come out narrower than the one counting
    40 clusters. At the top difficulty setting, where the slow test measures
    the widening at 17% on average; at ``difficulty_sd = 0`` the two widths are
    within a few percent and a single dataset can land either way.
    """
    (judge, human), cluster = clustered_ratings(
        np.random.default_rng(11), 40, 3, difficulty_sd=DIFFICULTY_LEVELS[-1]
    )

    naive = agree(judge, human, n_resamples=999, rng=12)
    clustered = agree(judge, human, cluster=cluster, n_resamples=999, rng=12)

    assert naive.ci_high - naive.ci_low < clustered.ci_high - clustered.ci_low
    assert naive.n_clusters == 120
    assert clustered.n_clusters == 40
    assert clustered.cluster_sizes == (3, 3, 3)


def test_j11_few_clusters_warns() -> None:
    """Ten clusters is below the floor, and the warning is not optional.

    A judge validated on ten recordings has an interval the resampling cannot
    support, and the warning says which way the error runs.

    **While the body is unimplemented this test reports "DID NOT WARN", and that
    message is not the failure.** ``pytest.warns`` catches the
    ``NotImplementedError``, finds no warning was recorded, and reports the
    absence rather than the exception; the traceback's "During handling of the
    above exception" is where the real cause is. A reader who takes the message
    at face value goes looking for a fault in the warning logic, which is not
    where the fault is. The same shape as the case in the retrospective where a
    red test did not distinguish two hypotheses: here a reported error hides the
    actual one.
    """
    (judge, human), cluster = clustered_ratings(np.random.default_rng(13), 10, 12)

    with pytest.warns(FewClustersWarning):
        agree(judge, human, cluster=cluster, n_resamples=199, rng=14)


def test_j11_run_records_its_own_seed() -> None:
    """A run reproduces after the fact, not only in advance.

    The seed is drawn and recorded even when the caller passed none, and
    rerunning at that seed returns the same interval.
    """
    (judge, human), cluster = clustered_ratings(np.random.default_rng(15), 40, 3)

    first = agree(judge, human, cluster=cluster, n_resamples=199)
    assert first.seed is not None

    again = agree(judge, human, cluster=cluster, n_resamples=199, rng=first.seed)
    assert (again.ci_low, again.ci_high) == (first.ci_low, first.ci_high)
    assert again.kappa == first.kappa


# ---------------------------------------------------------------------------
# J12  the ceiling comparison is paired, and it is an addition
# ---------------------------------------------------------------------------


def test_j12_ceiling_difference_comes_from_the_same_resamples() -> None:
    """The difference is a paired quantity, and its interval has to show it.

    Three raters on the same 120 items: the judge, the human it is scored
    against, and the second human that supplies the ceiling. Two claims, and the
    second is D7's first reason for making this an argument rather than a second
    function.

    The point estimate is the plain difference of the two coefficients. The
    interval is not the plain difference of the two intervals: both coefficients
    are computed on the same items and move together across resamples, so the
    paired interval is strictly narrower than the one built by treating them as
    independent. Building it the independent way would overstate the
    uncertainty in exactly the comparison the study turns on.
    """
    (judge, human, second), cluster = clustered_ratings(
        np.random.default_rng(21), 40, 3, n_raters=3
    )

    result = agree(
        judge, human, cluster=cluster, ceiling=second, n_resamples=999, rng=22
    )
    assert result.ceiling is not None

    assert result.ceiling.difference == result.kappa - result.ceiling.kappa

    paired_width = result.ceiling.difference_ci_high - result.ceiling.difference_ci_low
    independent_width = (result.ci_high - result.ceiling.ci_low) - (
        result.ci_low - result.ceiling.ci_high
    )
    assert paired_width < independent_width


@pytest.mark.parametrize(
    "rng",
    [22, np.random.default_rng(22)],
    ids=["seed", "generator"],
)
def test_j12_the_three_distributions_are_aligned_resample_by_resample(
    rng: int | np.random.Generator,
) -> None:
    """Check "the same resamples" on the output, not by reading the mechanism.

    If the judge's, the human's and the difference's distributions come from
    one set of draws, then resample by resample the difference is exactly the
    judge minus the human -- the same two floats, subtracted once. That
    equality is asserted element by element, which no narrowness comparison
    can stand in for: a difference statistic evaluated on its own draws is
    paired within itself and comes out narrow whether or not the other two
    passes saw the same indices.

    Both kinds of ``rng`` are run because they are the two cases a re-seeding
    design would have separated. With an integer seed a second pass re-seeded
    from it reproduces the first pass's draws; with a caller-supplied
    generator it continues the stream instead, and the three distributions
    silently stop lining up. A test written only with integer seeds would pass
    over exactly that fault. The draws are kept rather than re-seeded, so both
    cases have to hold, and both are pinned here.
    """
    (judge, human, second), cluster = clustered_ratings(
        np.random.default_rng(21), 40, 3, n_raters=3
    )

    result = agree(
        judge, human, cluster=cluster, ceiling=second, n_resamples=999, rng=rng
    )
    assert result.ceiling is not None

    assert result.distribution.size == result.n_valid
    assert result.ceiling.distribution.size == result.n_valid
    assert result.ceiling.difference_distribution.size == result.n_valid
    assert np.array_equal(
        result.distribution - result.ceiling.distribution,
        result.ceiling.difference_distribution,
    )


def test_j12_a_resample_undefined_for_either_pair_is_dropped_from_both() -> None:
    """One set of valid resamples serves every number in the result.

    The rule settled when the body was written: a resample on which either
    kappa is undefined is discarded from the judge's interval, the human's and
    the difference's alike, and ``n_valid`` counts the common set. The
    alternative -- discarding it from the difference only -- would leave the
    result carrying numbers from two different samplings, and nothing in the
    object would say which number came from which.

    The data is built so that only the human pair can degenerate. Thirty items,
    each its own cluster; the human and the second human agree exactly, and
    both use category 1 on four items and 0 on the rest, so a resample that
    misses all four leaves them constant on one shared category. The judge
    cycles through all three categories and is never constant, so the judge
    pair is defined on every resample: at the same seed, the call without
    ``ceiling`` discards nothing. The probability of missing all four is
    ``(26/30)**30``, about 1.4%, well under the floor that turns the warning
    into an error.
    """
    human = np.zeros(30, dtype=np.intp)
    human[[3, 11, 19, 27]] = 1
    second = human.copy()
    judge = np.tile(np.arange(3, dtype=np.intp), 10)

    alone = agree(judge, human, n_resamples=999, rng=31)
    assert alone.n_valid == alone.n_resamples

    with pytest.warns(DegenerateResampleWarning):
        result = agree(judge, human, ceiling=second, n_resamples=999, rng=31)
    assert result.ceiling is not None

    assert result.n_valid < result.n_resamples
    assert result.distribution.size == result.n_valid
    assert result.ceiling.distribution.size == result.n_valid
    assert result.ceiling.difference_distribution.size == result.n_valid
    assert np.array_equal(
        result.distribution - result.ceiling.distribution,
        result.ceiling.difference_distribution,
    )


def test_j12_the_ceiling_is_an_addition_not_a_modification() -> None:
    """Asking for the ceiling does not change the judge's own figure.

    At the same seed the draws are the same draws, so the judge's coefficient
    and interval have to come back identical whether or not a ceiling was asked
    for. Anything else would mean the reported agreement depends on what else
    was requested alongside it.

    The equality is asserted on a run where no resample was discarded, so that
    it pins the draws rather than prejudging how a discarded resample should be
    handled when two coefficients share it -- that rule is decided when the body
    is written.
    """
    (judge, human, second), cluster = clustered_ratings(
        np.random.default_rng(23), 40, 3, n_raters=3
    )

    alone = agree(judge, human, cluster=cluster, n_resamples=999, rng=24)
    with_ceiling = agree(
        judge, human, cluster=cluster, ceiling=second, n_resamples=999, rng=24
    )

    assert alone.n_valid == alone.n_resamples
    assert with_ceiling.n_valid == with_ceiling.n_resamples
    assert with_ceiling.kappa == alone.kappa
    assert (with_ceiling.ci_low, with_ceiling.ci_high) == (alone.ci_low, alone.ci_high)
    assert alone.ceiling is None
