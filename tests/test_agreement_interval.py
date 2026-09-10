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

from evalstat import FewClustersWarning
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
) -> tuple[list[NDArray[np.intp]], NDArray[np.intp]]:
    """Draw ordinal ratings that share a cluster-level component.

    Each item carries a latent quality ``a_i + e_ij`` with ``var(a) = RHO`` and
    ``var(e) = 1 - RHO``, so the latent variance is 1 whatever ``RHO`` is. Every
    rater sees that latent value through independent noise of size
    ``RATER_NOISE`` and cuts it into categories at ``CUTS``.

    Two things follow, and the test uses both. Raters agree because they are
    looking at the same latent value, not because they were built to agree; and
    the items of one cluster are alike, so an interval that resamples items
    rather than clusters counts information that is not there.

    The joint distribution of any two raters' labels does not depend on ``RHO``
    -- the cluster component is part of the latent variance either way -- so the
    population coefficient is a property of the generator alone, which is what
    makes it estimable once and reusable across designs.
    """
    latent = rng.normal(0.0, np.sqrt(RHO), size=(n_clusters, 1)) + rng.normal(
        0.0, np.sqrt(1.0 - RHO), size=(n_clusters, items_per_cluster)
    )
    raters = [
        np.digitize(
            (latent + rng.normal(0.0, RATER_NOISE, size=latent.shape)).ravel(), CUTS
        ).astype(np.intp)
        for _ in range(n_raters)
    ]
    cluster = np.repeat(np.arange(n_clusters), items_per_cluster).astype(np.intp)
    return raters, cluster


def population_kappa(weights: str = "linear") -> float:
    """Estimate the generator's own coefficient, with the outside witness.

    100_000 items from the same generator, scored by ``sklearn``. The Monte
    Carlo error on this is far below the width of any interval J11 tests, and
    the figure never passes through the code under test.

    As in ``test_agreement.py``, the weighting crosses as the string
    ``"linear"`` and the weight matrix is ``sklearn``'s own. A matrix built here
    would make the target of the coverage measurement depend on the code the
    measurement is about.
    """
    raters, _ = clustered_ratings(np.random.default_rng(90210), 50_000, 2)
    return float(
        cohen_kappa_score(
            raters[0], raters[1], labels=list(CATEGORIES), weights=weights
        )
    )


def agree(judge: NDArray[Any], human: NDArray[Any], **kwargs: Any) -> Any:
    """Call with this file's fixed scale and weighting."""
    return judge_agreement(
        judge, human, categories=CATEGORIES, weights="linear", **kwargs
    )


# ---------------------------------------------------------------------------
# J11  clustering restores coverage, and the run records itself
# ---------------------------------------------------------------------------

N_DATASETS = 200
SIM_RESAMPLES = 499


@pytest.mark.slow
def test_j11_cluster_resampling_restores_coverage() -> None:
    """Item-level resampling under-covers on clustered ratings; ``cluster=`` does not.

    The premise of the module in one measurement. Both halves matter: the naive
    interval has to be shown to under-cover, or there is nothing to correct, and
    the clustered one has to be shown to recover most of it. Neither is expected
    to reach nominal at forty clusters -- the bootstrap is consistent in the
    number of clusters and forty is not many -- and the assertion band says so.
    """
    truth = population_kappa()
    naive_covered = 0
    clustered_covered = 0
    for seed in range(N_DATASETS):
        (judge, human), cluster = clustered_ratings(
            np.random.default_rng(7000 + seed), 40, 3
        )
        naive = agree(judge, human, n_resamples=SIM_RESAMPLES, rng=8000 + seed)
        clustered = agree(
            judge, human, cluster=cluster, n_resamples=SIM_RESAMPLES, rng=8000 + seed
        )
        naive_covered += naive.ci_low <= truth <= naive.ci_high
        clustered_covered += clustered.ci_low <= truth <= clustered.ci_high

    naive_coverage = naive_covered / N_DATASETS
    clustered_coverage = clustered_covered / N_DATASETS

    assert naive_coverage < 0.91, (
        f"naive coverage {naive_coverage:.3f} did not under-cover, which is the "
        "whole premise of resampling clusters"
    )
    assert clustered_coverage > naive_coverage + 0.03
    assert 0.90 <= clustered_coverage <= 0.98


def test_j11_clustered_interval_is_wider_than_the_naive_one() -> None:
    """The fast half of the same claim, on one dataset rather than two hundred.

    Under-coverage and narrowness are the same fact seen from two sides, and
    this direction is cheap to check: the naive interval, counting 120 items as
    120 independent observations, has to come out narrower than the one counting
    40 clusters.
    """
    (judge, human), cluster = clustered_ratings(np.random.default_rng(11), 40, 3)

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
