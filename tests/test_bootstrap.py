"""Tests for :func:`evalstat.paired_bootstrap`.

The fast tests pin behaviour against results known independently of this code:
an analytic normal interval, the design-effect factor sqrt(1 + (m-1)rho), the
ratio 1 when there is no clustering to correct for, and exact equality with a
bootstrap over cluster means when within-cluster correlation is perfect.

The two simulation tests measure coverage and are marked slow.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from evalstat import (
    DegenerateResampleWarning,
    FewClustersWarning,
    PairedBootstrapResult,
    paired_bootstrap,
)
from evalstat.bootstrap import Method

METHODS: tuple[Method, ...] = ("percentile", "basic", "bca")


def clustered_differences(
    rng: np.random.Generator,
    n_clusters: int,
    items_per_cluster: int,
    mu: float,
    rho: float,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Draw differences with a known intra-cluster correlation.

    ``d_ij = mu + a_i + e_ij`` with ``var(a) = rho`` and ``var(e) = 1 - rho``,
    so the total variance is 1 and the intra-cluster correlation is ``rho``.
    """
    cluster_effect = rng.normal(0.0, np.sqrt(rho), size=n_clusters)
    noise = rng.normal(0.0, np.sqrt(1.0 - rho), size=(n_clusters, items_per_cluster))
    diff = (mu + cluster_effect[:, None] + noise).ravel()
    labels = np.repeat(np.arange(n_clusters), items_per_cluster)
    return diff, labels


def preference_rate(d: NDArray[np.float64]) -> float:
    """Share of non-tied items favouring the first system.

    Returns NaN rather than dividing by zero when a resample is all ties; the
    caller counts those and decides whether too many have accumulated.
    """
    decided = int(np.count_nonzero(d != 0))
    if decided == 0:
        return float("nan")
    return float(np.count_nonzero(d > 0)) / decided


def width(result: PairedBootstrapResult) -> float:
    return result.ci_high - result.ci_low


# --------------------------------------------------------------------------
# T1  cluster=None is the independent case by construction, not by analogy
# --------------------------------------------------------------------------


def test_cluster_none_equals_one_cluster_per_item() -> None:
    rng = np.random.default_rng(0)
    diff = rng.normal(0.4, 1.0, size=200)

    unclustered = paired_bootstrap(diff, n_resamples=500, rng=11)
    singletons = paired_bootstrap(
        diff, cluster=np.arange(diff.size), n_resamples=500, rng=11
    )

    assert unclustered.estimate == singletons.estimate
    assert unclustered.ci_low == singletons.ci_low
    assert unclustered.ci_high == singletons.ci_high
    assert unclustered.n_clusters == singletons.n_clusters == diff.size
    np.testing.assert_array_equal(unclustered.distribution, singletons.distribution)


# --------------------------------------------------------------------------
# T2  unclustered interval matches the analytic normal interval
# --------------------------------------------------------------------------


def test_unclustered_interval_matches_analytic_normal_interval() -> None:
    n = 4000
    sigma = 1.0
    rng = np.random.default_rng(7)
    diff = rng.normal(0.5, sigma, size=n)

    result = paired_bootstrap(diff, n_resamples=4000, rng=3)

    half_width = 1.959963985 * diff.std(ddof=1) / np.sqrt(n)
    assert result.ci_low == pytest.approx(diff.mean() - half_width, abs=0.005)
    assert result.ci_high == pytest.approx(diff.mean() + half_width, abs=0.005)


# --------------------------------------------------------------------------
# T3  seeded and reproducible
# --------------------------------------------------------------------------


def test_same_seed_reproduces_and_different_seed_does_not() -> None:
    diff, labels = clustered_differences(
        np.random.default_rng(1), 40, 3, mu=0.3, rho=0.2
    )

    first = paired_bootstrap(diff, cluster=labels, n_resamples=400, rng=99)
    again = paired_bootstrap(diff, cluster=labels, n_resamples=400, rng=99)
    other = paired_bootstrap(diff, cluster=labels, n_resamples=400, rng=100)

    assert first.seed == again.seed == 99
    np.testing.assert_array_equal(first.distribution, again.distribution)
    assert first.ci_low != other.ci_low


def test_unseeded_call_records_a_usable_seed() -> None:
    diff, labels = clustered_differences(
        np.random.default_rng(2), 40, 3, mu=0.3, rho=0.2
    )

    spontaneous = paired_bootstrap(diff, cluster=labels, n_resamples=300)
    assert spontaneous.seed is not None

    replayed = paired_bootstrap(
        diff, cluster=labels, n_resamples=300, rng=spontaneous.seed
    )
    np.testing.assert_array_equal(spontaneous.distribution, replayed.distribution)


def test_caller_supplied_generator_records_no_seed() -> None:
    diff = np.random.default_rng(3).normal(0.2, 1.0, size=90)
    result = paired_bootstrap(diff, n_resamples=200, rng=np.random.default_rng(5))
    assert result.seed is None


# --------------------------------------------------------------------------
# T4  clustered intervals are wider by the design effect, not merely wider
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.1, 0.2])
def test_clustered_interval_is_wider_by_the_design_effect(rho: float) -> None:
    n_clusters, items = 40, 3
    expected = np.sqrt(1 + (items - 1) * rho)

    ratios = []
    for seed in range(40):
        diff, labels = clustered_differences(
            np.random.default_rng(seed), n_clusters, items, mu=0.3, rho=rho
        )
        clustered = paired_bootstrap(diff, cluster=labels, n_resamples=800, rng=seed)
        naive = paired_bootstrap(diff, n_resamples=800, rng=seed)
        ratios.append(width(clustered) / width(naive))

    assert np.mean(ratios) == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------
# T5  and no wider when there is nothing to correct for
# --------------------------------------------------------------------------


def test_no_inflation_when_there_is_no_clustering() -> None:
    ratios = []
    for seed in range(40):
        diff, labels = clustered_differences(
            np.random.default_rng(seed), 40, 3, mu=0.3, rho=0.0
        )
        clustered = paired_bootstrap(diff, cluster=labels, n_resamples=800, rng=seed)
        naive = paired_bootstrap(diff, n_resamples=800, rng=seed)
        ratios.append(width(clustered) / width(naive))

    assert np.mean(ratios) == pytest.approx(1.0, rel=0.05)


# --------------------------------------------------------------------------
# T6  perfect within-cluster correlation reduces to the cluster means
# --------------------------------------------------------------------------


def test_perfectly_correlated_clusters_match_bootstrap_of_cluster_means() -> None:
    n_clusters, items = 40, 3
    values = np.random.default_rng(4).normal(0.5, 1.0, size=n_clusters)
    diff = np.repeat(values, items)
    labels = np.repeat(np.arange(n_clusters), items)

    clustered = paired_bootstrap(diff, cluster=labels, n_resamples=2000, rng=21)
    on_means = paired_bootstrap(values, n_resamples=2000, rng=21)

    assert clustered.estimate == pytest.approx(on_means.estimate)
    assert clustered.ci_low == pytest.approx(on_means.ci_low)
    assert clustered.ci_high == pytest.approx(on_means.ci_high)


# --------------------------------------------------------------------------
# T7  the preference-rate statistic, against a known generating probability
# --------------------------------------------------------------------------


def test_preference_rate_recovers_its_generating_probability() -> None:
    rng = np.random.default_rng(6)
    probabilities = [0.6, 0.3, 0.1]  # B better, tie, A better
    truth = 0.6 / (0.6 + 0.1)
    diff = rng.choice([1.0, 0.0, -1.0], size=1200, p=probabilities)

    result = paired_bootstrap(diff, statistic=preference_rate, n_resamples=2000, rng=13)

    assert result.estimate == pytest.approx(truth, abs=0.05)
    assert result.ci_low < truth < result.ci_high


# --------------------------------------------------------------------------
# T8  degenerate resamples: discard, count, warn, then refuse
# --------------------------------------------------------------------------


def test_a_few_all_tie_resamples_are_discarded_with_a_warning() -> None:
    diff = np.zeros(120)
    diff[::30] = 1.0  # four decided items in forty clusters of three
    labels = np.repeat(np.arange(40), 3)

    with pytest.warns(DegenerateResampleWarning):
        result = paired_bootstrap(
            diff,
            cluster=labels,
            statistic=preference_rate,
            n_resamples=2000,
            rng=17,
        )

    assert result.n_valid < result.n_resamples
    assert result.n_valid >= 0.95 * result.n_resamples
    assert result.distribution.size == result.n_valid


def test_too_many_degenerate_resamples_is_an_error() -> None:
    diff = np.zeros(120)
    diff[0] = 1.0  # a single decided item: most resamples miss it entirely
    labels = np.repeat(np.arange(40), 3)

    with pytest.raises(ValueError, match="conditional quantity"):
        paired_bootstrap(
            diff,
            cluster=labels,
            statistic=preference_rate,
            n_resamples=2000,
            rng=19,
        )


# --------------------------------------------------------------------------
# T9  input validation
# --------------------------------------------------------------------------


def test_mismatched_pair_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap(np.zeros(10), np.zeros(9))


def test_non_finite_differences_are_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        paired_bootstrap(np.array([1.0, np.nan, 3.0]))


def test_wrong_cluster_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="one label per pair"):
        paired_bootstrap(np.zeros(10), cluster=np.zeros(9))


def test_a_single_cluster_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        paired_bootstrap(np.arange(30.0), cluster=np.zeros(30))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": 0}, "n_resamples must be positive"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"method": "studentized"}, "unknown method"),
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(np.arange(30.0), **kwargs)


def test_few_clusters_warns_about_the_direction_of_the_error() -> None:
    diff, labels = clustered_differences(
        np.random.default_rng(8), 10, 3, mu=0.3, rho=0.2
    )
    with pytest.warns(FewClustersWarning, match="narrower"):
        paired_bootstrap(diff, cluster=labels, n_resamples=200, rng=23)


# --------------------------------------------------------------------------
# T10  unequal cluster sizes
# --------------------------------------------------------------------------


def test_unequal_cluster_sizes_are_handled() -> None:
    sizes = [1, 2, 5] * 10
    labels = np.repeat(np.arange(len(sizes)), sizes)
    diff = np.random.default_rng(9).normal(0.4, 1.0, size=labels.size)

    result = paired_bootstrap(diff, cluster=labels, n_resamples=500, rng=29)

    assert result.n_clusters == len(sizes)
    assert result.cluster_sizes == (1, 2, 5)
    assert result.n_pairs == labels.size


# --------------------------------------------------------------------------
# T11  sign convention
# --------------------------------------------------------------------------


def test_swapping_the_arguments_mirrors_the_interval() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(1.0, 1.0, size=150)
    y = rng.normal(0.0, 1.0, size=150)

    forward = paired_bootstrap(x, y, n_resamples=1000, rng=31)
    backward = paired_bootstrap(y, x, n_resamples=1000, rng=31)

    assert forward.estimate == pytest.approx(-backward.estimate)
    assert forward.ci_low == pytest.approx(-backward.ci_high)
    assert forward.ci_high == pytest.approx(-backward.ci_low)


def test_y_none_matches_passing_the_difference_explicitly() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(1.0, 1.0, size=150)
    y = rng.normal(0.0, 1.0, size=150)

    from_pair = paired_bootstrap(x, y, n_resamples=600, rng=37)
    from_diff = paired_bootstrap(x - y, n_resamples=600, rng=37)

    assert from_pair.ci_low == pytest.approx(from_diff.ci_low)
    assert from_pair.ci_high == pytest.approx(from_diff.ci_high)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_brackets_the_estimate(method: Method) -> None:
    diff, labels = clustered_differences(
        np.random.default_rng(14), 40, 3, mu=0.5, rho=0.2
    )
    result = paired_bootstrap(
        diff, cluster=labels, n_resamples=1500, rng=41, method=method
    )
    assert result.ci_low < result.estimate < result.ci_high
    assert result.method == method


# --------------------------------------------------------------------------
# T12, T13  coverage simulations
# --------------------------------------------------------------------------

N_DATASETS = 300
SIM_RESAMPLES = 499


@pytest.mark.slow
def test_unclustered_coverage_is_nominal() -> None:
    mu = 0.5
    covered = 0
    for seed in range(N_DATASETS):
        rng = np.random.default_rng(1000 + seed)
        diff = rng.normal(mu, 1.0, size=120)
        result = paired_bootstrap(diff, n_resamples=SIM_RESAMPLES, rng=2000 + seed)
        covered += result.ci_low <= mu <= result.ci_high

    assert 0.92 <= covered / N_DATASETS <= 0.98


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_cluster_resampling_restores_coverage(method: Method) -> None:
    """Naive resampling under-covers at rho = 0.2; cluster resampling does not.

    The measured figures are recorded in the module docstring of
    ``evalstat.bootstrap`` and are what the default interval method rests on.
    """
    mu, rho = 0.5, 0.2
    naive_covered = 0
    clustered_covered = 0
    for seed in range(N_DATASETS):
        diff, labels = clustered_differences(
            np.random.default_rng(3000 + seed), 40, 3, mu=mu, rho=rho
        )
        naive = paired_bootstrap(
            diff, n_resamples=SIM_RESAMPLES, rng=4000 + seed, method=method
        )
        clustered = paired_bootstrap(
            diff,
            cluster=labels,
            n_resamples=SIM_RESAMPLES,
            rng=4000 + seed,
            method=method,
        )
        naive_covered += naive.ci_low <= mu <= naive.ci_high
        clustered_covered += clustered.ci_low <= mu <= clustered.ci_high

    naive_coverage = naive_covered / N_DATASETS
    clustered_coverage = clustered_covered / N_DATASETS

    assert naive_coverage < 0.91, (
        f"{method}: naive coverage {naive_coverage:.3f} did not under-cover, "
        "which is the whole premise of cluster resampling"
    )
    assert clustered_coverage > naive_coverage + 0.03
    assert 0.90 <= clustered_coverage <= 0.98
