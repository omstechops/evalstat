"""Bootstrap confidence intervals for paired differences, resampled by cluster.

Evaluation items are rarely independent. Three items cut from one recording, or
three drawn from one article, share topic, speaker and acoustic state, so an
interval computed as though they were 120 independent observations is too narrow
and the significance it reports is inflated. The fix is to resample the *cluster*
rather than the item, which is what :func:`paired_bootstrap` does when it is
given ``cluster=``.

``scipy.stats.bootstrap`` already handles the paired case; it has no notion of a
cluster. That gap is the reason this module exists.

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

import secrets
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

__all__ = [
    "MIN_CLUSTERS",
    "MIN_VALID_FRACTION",
    "DegenerateResampleWarning",
    "FewClustersWarning",
    "PairedBootstrapResult",
    "paired_bootstrap",
]

MIN_CLUSTERS = 25
"""Below this many clusters, :func:`paired_bootstrap` warns rather than fails."""

MIN_VALID_FRACTION = 0.95
"""Below this share of usable resamples, :func:`paired_bootstrap` raises."""

Method = Literal["percentile", "basic", "bca"]


class FewClustersWarning(UserWarning):
    """Too few clusters for the resampling to be trusted at face value.

    The cluster bootstrap is consistent in the number of *clusters*, not in the
    number of items, so adding items to existing clusters does not silence this.
    """


class DegenerateResampleWarning(UserWarning):
    """Some resamples produced no finite value and were discarded.

    Raised as a warning while the discarded share stays small; past
    :data:`MIN_VALID_FRACTION` it becomes an error, because the quantity being
    reported has by then quietly become a conditional one.
    """


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


def _cluster_layout(
    cluster: ArrayLike | None, n_pairs: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Lay clusters out contiguously for indexing.

    Returns ``(members, starts, sizes)`` where ``members`` holds every item index
    grouped by cluster, and cluster ``c`` occupies
    ``members[starts[c] : starts[c] + sizes[c]]``.

    ``cluster=None`` is not a separate code path: it is expressed as one cluster
    per item, so the independent case is standard bootstrap by construction
    rather than by a second implementation that has to be kept in step.
    """
    if cluster is None:
        labels = np.arange(n_pairs)
    else:
        labels = np.asarray(cluster)
        if labels.shape != (n_pairs,):
            raise ValueError(
                f"cluster must have one label per pair; expected shape "
                f"({n_pairs},), got {labels.shape}"
            )
    _, codes = np.unique(labels, return_inverse=True)
    codes = codes.ravel()
    members = np.argsort(codes, kind="stable").astype(np.intp)
    sizes = np.bincount(codes).astype(np.intp)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.intp)
    return members, starts, sizes


def _draw_indices(
    rng: np.random.Generator,
    members: NDArray[np.intp],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Return the item indices of one with-replacement draw of whole clusters.

    Clusters are drawn to the observed count, and every item of a drawn cluster
    comes with it, so the resample size varies when clusters differ in size.

    The index arithmetic replaces a per-cluster Python loop. ``ramp`` is the
    position of each output slot *within* its own cluster: subtracting the
    repeated output offsets from a plain arange restarts the count at every
    cluster boundary, and adding the repeated source offsets then points each
    slot at the right member.
    """
    drawn = rng.integers(0, sizes.size, size=sizes.size)
    drawn_sizes = sizes[drawn]
    total = int(drawn_sizes.sum())
    out_starts = np.concatenate(([0], np.cumsum(drawn_sizes)[:-1]))
    ramp = np.arange(total) - np.repeat(out_starts, drawn_sizes)
    picked: NDArray[np.intp] = members[np.repeat(starts[drawn], drawn_sizes) + ramp]
    return picked


def _percentile_interval(
    distribution: NDArray[np.float64], alpha: float
) -> tuple[float, float]:
    """Interval from the bootstrap quantiles themselves."""
    low, high = np.quantile(distribution, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _basic_interval(
    distribution: NDArray[np.float64], estimate: float, alpha: float
) -> tuple[float, float]:
    """Interval reflected through the observed estimate."""
    low, high = np.quantile(distribution, [alpha / 2, 1 - alpha / 2])
    return float(2 * estimate - high), float(2 * estimate - low)


def _bca_interval(
    distribution: NDArray[np.float64],
    estimate: float,
    alpha: float,
    statistic: Callable[[NDArray[np.float64]], float],
    diff: NDArray[np.float64],
    members: NDArray[np.intp],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval, jackknifed over clusters.

    The jackknife leaves out a whole cluster at a time, matching the unit the
    bootstrap resamples. Leaving out an item instead would estimate the
    acceleration from a dependence structure the interval does not use.
    """
    n_valid = distribution.size
    below = float(np.count_nonzero(distribution < estimate))
    # Keep the proportion off 0 and 1 so the normal quantile stays finite when
    # the bootstrap distribution sits entirely to one side of the estimate.
    guard = 1.0 / (2.0 * n_valid)
    prop = min(max(below / n_valid, guard), 1.0 - guard)
    z0 = float(norm.ppf(prop))

    n_clusters = sizes.size
    jack = np.empty(n_clusters, dtype=np.float64)
    keep = np.ones(diff.size, dtype=bool)
    for c in range(n_clusters):
        block = members[starts[c] : starts[c] + sizes[c]]
        keep[block] = False
        jack[c] = statistic(diff[keep])
        keep[block] = True
    if not np.all(np.isfinite(jack)):
        raise ValueError(
            "leaving out a cluster made the statistic undefined, so the BCa "
            "acceleration cannot be estimated; use method='percentile'"
        )
    centred = jack.mean() - jack
    denominator = 6.0 * float(np.sum(centred**2)) ** 1.5
    accel = 0.0 if denominator == 0 else float(np.sum(centred**3)) / denominator

    z_low, z_high = norm.ppf([alpha / 2, 1 - alpha / 2])
    adjusted = []
    for z in (z_low, z_high):
        shifted = z0 + z
        adjusted.append(float(norm.cdf(z0 + shifted / (1 - accel * shifted))))
    low, high = np.quantile(distribution, adjusted)
    return float(low), float(high)


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
    if method not in ("percentile", "basic", "bca"):
        raise ValueError(f"unknown method {method!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")

    diff = _as_differences(x, y)
    members, starts, sizes = _cluster_layout(cluster, diff.size)
    n_clusters = int(sizes.size)
    if n_clusters < 2:
        raise ValueError(f"need at least 2 clusters to resample, got {n_clusters}")
    if n_clusters < MIN_CLUSTERS:
        warnings.warn(
            f"{n_clusters} clusters is below {MIN_CLUSTERS}; the cluster "
            "bootstrap is consistent in the number of clusters, and at this "
            "count the percentile interval under-covers — the interval it "
            "reports is narrower than the truth, not wider",
            FewClustersWarning,
            stacklevel=2,
        )

    if isinstance(rng, np.random.Generator):
        generator, seed = rng, None
    else:
        seed = secrets.randbits(64) if rng is None else int(rng)
        generator = np.random.default_rng(seed)

    estimate = float(statistic(diff))
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = _draw_indices(generator, members, starts, sizes)
        values[i] = statistic(diff[idx])

    finite = np.isfinite(values)
    distribution = values[finite]
    n_valid = int(distribution.size)
    if n_valid < MIN_VALID_FRACTION * n_resamples:
        raise ValueError(
            f"only {n_valid} of {n_resamples} resamples produced a finite "
            f"statistic, below the {MIN_VALID_FRACTION:.0%} floor. What is "
            "left is a conditional quantity — the statistic among resamples "
            "where it happened to be defined — and returning it as though it "
            "were the requested one would be wrong"
        )
    if n_valid < n_resamples:
        warnings.warn(
            f"{n_resamples - n_valid} of {n_resamples} resamples produced no "
            "finite statistic and were discarded",
            DegenerateResampleWarning,
            stacklevel=2,
        )

    alpha = 1.0 - confidence_level
    if method == "percentile":
        ci_low, ci_high = _percentile_interval(distribution, alpha)
    elif method == "basic":
        ci_low, ci_high = _basic_interval(distribution, estimate, alpha)
    else:
        ci_low, ci_high = _bca_interval(
            distribution,
            estimate,
            alpha,
            statistic,
            diff,
            members,
            starts,
            sizes,
        )

    return PairedBootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        method=method,
        n_pairs=int(diff.size),
        n_clusters=n_clusters,
        cluster_sizes=(
            int(sizes.min()),
            int(np.median(sizes)),
            int(sizes.max()),
        ),
        n_resamples=n_resamples,
        n_valid=n_valid,
        seed=seed,
        distribution=distribution,
    )
