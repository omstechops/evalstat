"""Bootstrap confidence intervals for paired differences, resampled by cluster.

Evaluation items are rarely independent. Three items cut from one recording, or
three drawn from one article, share topic, speaker and acoustic state, so an
interval computed as though they were 120 independent observations is too narrow
and the significance it reports is inflated. The fix is to resample the *cluster*
rather than the item, which is what :func:`paired_bootstrap` does when it is
given ``cluster=``.

``scipy.stats.bootstrap`` already handles the paired case; it has no notion of a
cluster. That gap is the reason this module exists.

The drawing itself is not here. It lives in :mod:`evalstat._resample`, keyed on
indices, because the same cluster resampling carries every interval this package
reports and only the statistic's argument contract differs between them. This
module owns that contract for the paired case -- one vector of differences --
and nothing else about the resampling.

Documented coverage finding
---------------------------
Empirical coverage of nominal 95% intervals, measured by the simulation in
``tests/test_bootstrap.py::test_cluster_resampling_restores_coverage``: 300
datasets at k = 40 clusters, m = 3 items each, intra-cluster correlation
rho = 0.2, mean of the paired differences, 499 resamples per interval.

============  ==================  ====================
Method        Item-level (naive)  Cluster-level
============  ==================  ====================
percentile    0.887               0.927
basic         0.890               0.933
bca           0.883               0.923
============  ==================  ====================

Monte Carlo standard error on each figure is about 0.015, so read differences
below three points as noise. Three things follow.

Naive resampling under-covers by roughly six points at rho = 0.2. That is the
whole case for ``cluster=``, and it is a measured case rather than an argued one.

Cluster resampling recovers most of it but **does not reach nominal**: 0.92-0.93
against a nominal 0.95. This is assumption 3 in :func:`paired_bootstrap` showing
up in the numbers — consistency is in the number of clusters, and forty clusters
is not many. An interval from this function at k = 40 is still somewhat too
narrow, and a study reporting one should say so rather than treat 95% as
delivered.

The three interval methods are indistinguishable at this cluster count; BCa is
if anything marginally the worst, its acceleration term being estimated from the
same forty clusters. So the default is ``percentile``, the method with the
fewest estimated quantities: preferring BCa here would mean adding an unstable
correction for no measured gain. This table is a measurement, not a prediction,
and the default is open to revision if the measurement changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evalstat._resample import (
    MIN_CLUSTERS,
    MIN_VALID_FRACTION,
    DegenerateResampleWarning,
    FewClustersWarning,
    check_resampling_arguments,
    cluster_bootstrap,
)

# Re-exported in the alias form because it is not in __all__, and __all__ is
# the curated public list rather than a record of what happens to be importable.
from evalstat._resample import Method as Method

__all__ = [
    "MIN_CLUSTERS",
    "MIN_VALID_FRACTION",
    "DegenerateResampleWarning",
    "FewClustersWarning",
    "PairedBootstrapResult",
    "paired_bootstrap",
]


@dataclass(frozen=True)
class PairedBootstrapResult:
    """One bootstrap interval, with the record needed to reproduce it.

    Attributes
    ----------
    estimate
        ``statistic`` applied to the observed differences.
    ci_low, ci_high
        Interval endpoints.
    confidence_level
        Nominal coverage requested.
    method
        Interval construction used: ``percentile``, ``basic`` or ``bca``.
    n_pairs
        Number of paired observations.
    n_clusters
        Number of resampling units. Equals ``n_pairs`` when ``cluster`` is None.
    cluster_sizes
        ``(minimum, median, maximum)`` items per cluster.
    n_resamples
        Resamples requested.
    n_valid
        Resamples that yielded a finite statistic and entered the interval.
    seed
        Seed the resampling ran from. Populated even when the caller passed
        nothing, so a run is reproducible after the fact; ``None`` only when the
        caller supplied its own :class:`numpy.random.Generator`, whose state this
        function cannot record.
    distribution
        The ``n_valid`` resampled statistics, for onward use such as a minimum
        detectable effect curve.
    """

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    method: str
    n_pairs: int
    n_clusters: int
    cluster_sizes: tuple[int, int, int]
    n_resamples: int
    n_valid: int
    seed: int | None
    distribution: NDArray[np.float64]


def _as_differences(x: ArrayLike, y: ArrayLike | None) -> NDArray[np.float64]:
    """Return the paired differences ``x - y``, validated."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x_arr.shape}")
    if y is None:
        diff = x_arr
    else:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.shape != x_arr.shape:
            raise ValueError(
                f"x and y must have the same shape; got {x_arr.shape} and "
                f"{y_arr.shape}. Pairing must be complete: a system that "
                "dropped an item breaks the design, it does not shorten it."
            )
        diff = x_arr - y_arr
    if diff.size == 0:
        raise ValueError("need at least one pair")
    if not np.all(np.isfinite(diff)):
        raise ValueError(
            "differences contain NaN or infinity; missing pairs must be "
            "resolved before resampling, not dropped inside it"
        )
    return diff


def paired_bootstrap(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    cluster: ArrayLike | None = None,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    method: Method = "percentile",
    rng: int | np.random.Generator | None = None,
) -> PairedBootstrapResult:
    """Bootstrap the paired difference ``x - y``; positive favours ``x``.

    The sign convention is the first thing stated because it is the easiest
    thing to get backwards and the hardest to notice afterwards: an interval
    reported with the arguments swapped is a correct interval for the opposite
    conclusion.

    Parameters
    ----------
    x, y
        Paired observations of equal length, aligned by position. Pass ``y=None``
        when ``x`` already holds the differences — preference data recorded as
        -1, 0, +1 arrives that way.
    cluster
        One label per pair. Items sharing a label are resampled together, as a
        unit. ``None`` means every item is its own cluster, which is the
        ordinary independent bootstrap.
    statistic
        Applied to the vector of differences and returning one number. Defaults
        to the mean. A preference rate is
        ``lambda d: (d > 0).sum() / (d != 0).sum()``.

        This deliberately takes the differences, not ``x`` and ``y``
        separately, so quantities of the form ``statistic(x) - statistic(y)``
        — the difference of two medians, say — are out of scope. Their
        resampling logic is not this one, and hosting two resampling logics in
        one function is a way of getting one of them wrong.
    n_resamples
        Cluster draws to take.
    confidence_level
        Nominal coverage.
    method
        ``percentile`` (default), ``basic`` or ``bca``. See the module docstring
        for the measured coverage of each at this study's cluster count.
    rng
        Seed, generator, or ``None``. When ``None`` a seed is drawn and recorded
        on the result, so a run stays reproducible even if nobody thought to
        pass one.

    Returns
    -------
    PairedBootstrapResult
        Estimate, interval, and the provenance needed to reproduce it.

    Raises
    ------
    ValueError
        If the pairing is incomplete, the clustering is malformed, fewer than
        two clusters are present, or more than ``1 - MIN_VALID_FRACTION`` of
        resamples failed to produce a finite statistic.

    Warns
    -----
    FewClustersWarning
        Fewer than :data:`MIN_CLUSTERS` clusters.
    DegenerateResampleWarning
        Some, but not too many, resamples were discarded.

    Notes
    -----
    Assumptions, and where each one fails:

    1. **Clusters are mutually independent.** Dependence *within* a cluster may
       take any form — that is the point. Two articles by one author, or two
       recordings of one speaker, treated as separate clusters, break this. So
       does a single synthesised voice shared by every cluster: the voice's
       contribution is not merely unestimated, it is absent from the material,
       and the interval does not license generalising to other voices.
    2. **Clusters are identically distributed.** Pooling arms of a study, or
       pooling domains of very different difficulty without stratifying, breaks
       this.
    3. **Consistency is in the number of clusters, not items.** Coverage decays
       as clusters become few, and adding items to existing clusters does not
       help. See :data:`MIN_CLUSTERS`.
    4. **The statistic is a smooth functional of the empirical distribution.**
       Medians and quantiles converge slowly here; ratio statistics are
       undefined when a resample empties the denominator.
    5. **Pairing is complete and exact.** One system dropping an item is a
       design failure, not a shorter array.
    6. **The percentile interval assumes an approximately symmetric bootstrap
       distribution.** ``bca`` corrects skew, at the price of an estimated
       acceleration that is itself unstable when clusters are few.
    7. **Unequal cluster sizes make the resample size random**, which biases
       ratio statistics slightly. Equal sizes are not assumed.
    8. **This returns an interval, not a test.** An interval excluding zero is
       not a cluster permutation test, carries no multiplicity correction, and
       applies no small-cluster correction: the wild cluster bootstrap is *not
       implemented* here.
    9. **The cluster labels are taken to be correct.** Mis-specified clustering
       cannot be detected from the inside; item-level labels silently return the
       naive interval.
    """
    # Ahead of _as_differences so that a malformed n_resamples is reported as
    # such whether or not the data is also malformed; cluster_bootstrap checks
    # the same three again, from the same definition.
    check_resampling_arguments(n_resamples, confidence_level, method)
    diff = _as_differences(x, y)

    drawn = cluster_bootstrap(
        lambda idx: float(statistic(diff[idx])),
        int(diff.size),
        cluster=cluster,
        unit="pair",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        rng=rng,
    )
    return PairedBootstrapResult(
        estimate=drawn.estimate,
        ci_low=drawn.ci_low,
        ci_high=drawn.ci_high,
        confidence_level=confidence_level,
        method=method,
        n_pairs=int(diff.size),
        n_clusters=drawn.n_clusters,
        cluster_sizes=drawn.cluster_sizes,
        n_resamples=n_resamples,
        n_valid=drawn.n_valid,
        seed=drawn.seed,
        distribution=drawn.distribution,
    )
