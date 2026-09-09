"""Cluster resampling, shared by every interval this package reports.

Resampling whole clusters rather than items is the one correction this package
is built around, and it is the same correction whatever statistic sits on top of
it. What differs between statistics is not how the resample is drawn but what
they are handed once it is: :func:`evalstat.paired_bootstrap` wants one vector
of paired differences, an agreement coefficient wants two parallel vectors of
labels, and a difference of differences would want more still.

So the resampler here is keyed on **indices**. ``statistic`` receives the item
indices of one resample and returns a number; what it indexes into is its own
business. That keeps one drawing implementation under every public face while
leaving each face free to define its own argument contract.

The alternative was to widen :func:`evalstat.paired_bootstrap` until it accepted
both contracts, and its docstring already rules that out: hosting two resampling
logics in one function is a way of getting one of them wrong. This module is the
other reading of that sentence. There is one resampling logic, and it lives
here; the two contracts stay apart, above it.

Nothing in this module is public. The names re-exported from
:mod:`evalstat.bootstrap` -- :data:`MIN_CLUSTERS`, :data:`MIN_VALID_FRACTION`,
:class:`FewClustersWarning`, :class:`DegenerateResampleWarning` -- are the public
spelling, and they describe the cluster bootstrap generally rather than any one
statistic built on it.
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
    "ClusterBootstrapResult",
    "DegenerateResampleWarning",
    "FewClustersWarning",
    "Method",
    "cluster_bootstrap",
]

MIN_CLUSTERS = 25
"""Below this many clusters, the cluster bootstrap warns rather than fails."""

MIN_VALID_FRACTION = 0.95
"""Below this share of usable resamples, the cluster bootstrap raises."""

Method = Literal["percentile", "basic", "bca"]

Statistic = Callable[[NDArray[np.intp]], float]
"""Applied to the item indices of one resample, returning one number.

Indices rather than data: see the module docstring. A statistic that needs two
aligned vectors closes over both and indexes each, which is exactly what an
agreement coefficient does and what a single vector of differences cannot
express.
"""


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
class ClusterBootstrapResult:
    """One interval and the resampling record behind it.

    Deliberately narrower than the results the public functions return: it
    carries what the resampling knows and nothing about what was resampled.
    Each public face adds its own vocabulary on top -- how many pairs, which
    categories, what the statistic was -- because those are facts about the
    statistic, not about the draw.

    Attributes
    ----------
    estimate
        ``statistic`` applied to every item once.
    ci_low, ci_high
        Interval endpoints.
    n_clusters
        Number of resampling units.
    cluster_sizes
        ``(minimum, median, maximum)`` items per cluster.
    n_valid
        Resamples that yielded a finite statistic and entered the interval.
    seed
        Seed the resampling ran from, or ``None`` when the caller supplied its
        own generator, whose state cannot be recorded.
    distribution
        The ``n_valid`` resampled statistics.
    """

    estimate: float
    ci_low: float
    ci_high: float
    n_clusters: int
    cluster_sizes: tuple[int, int, int]
    n_valid: int
    seed: int | None
    distribution: NDArray[np.float64]


def check_resampling_arguments(
    n_resamples: int, confidence_level: float, method: Method
) -> None:
    """Validate the three arguments that describe the draw itself.

    Split out so that a public function can run these before touching the
    caller's data, which keeps the error reported for a malformed
    ``n_resamples`` independent of whether the data is also malformed.
    :func:`cluster_bootstrap` calls it too, being an entry point in its own
    right; the checks are pure, so running them twice costs nothing and having
    one definition means the messages cannot drift apart.
    """
    if method not in ("percentile", "basic", "bca"):
        raise ValueError(f"unknown method {method!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")


def _cluster_layout(
    cluster: ArrayLike | None, n_items: int, unit: str
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Lay clusters out contiguously for indexing.

    Returns ``(members, starts, sizes)`` where ``members`` holds every item index
    grouped by cluster, and cluster ``c`` occupies
    ``members[starts[c] : starts[c] + sizes[c]]``.

    ``cluster=None`` is not a separate code path: it is expressed as one cluster
    per item, so the independent case is standard bootstrap by construction
    rather than by a second implementation that has to be kept in step.

    ``unit`` is the caller's word for one row -- a pair of ratings, an item, a
    recording -- and appears only in the error message, so that each public face
    reports the shape mismatch in its own vocabulary.
    """
    if cluster is None:
        labels = np.arange(n_items)
    else:
        labels = np.asarray(cluster)
        if labels.shape != (n_items,):
            raise ValueError(
                f"cluster must have one label per {unit}; expected shape "
                f"({n_items},), got {labels.shape}"
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
    statistic: Statistic,
    n_items: int,
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
    keep = np.ones(n_items, dtype=bool)
    for c in range(n_clusters):
        block = members[starts[c] : starts[c] + sizes[c]]
        keep[block] = False
        jack[c] = statistic(np.flatnonzero(keep).astype(np.intp))
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


def cluster_bootstrap(
    statistic: Statistic,
    n_items: int,
    *,
    cluster: ArrayLike | None = None,
    unit: str = "item",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    method: Method = "percentile",
    rng: int | np.random.Generator | None = None,
) -> ClusterBootstrapResult:
    """Bootstrap ``statistic`` by resampling whole clusters with replacement.

    Parameters
    ----------
    statistic
        Applied to the item indices of one resample; see :data:`Statistic`.
    n_items
        Number of rows the indices range over.
    cluster
        One label per item. Items sharing a label are resampled together, as a
        unit. ``None`` means every item is its own cluster.
    unit
        The caller's word for one row, used only in the shape error message.
    n_resamples, confidence_level, method, rng
        As on the public functions built over this.

    Returns
    -------
    ClusterBootstrapResult

    Raises
    ------
    ValueError
        If the resampling arguments are malformed, the clustering does not match
        ``n_items``, fewer than two clusters are present, or more than
        ``1 - MIN_VALID_FRACTION`` of resamples failed to produce a finite
        statistic.

    Warns
    -----
    FewClustersWarning, DegenerateResampleWarning
        Both at ``stacklevel=3``, so they surface at whoever called the public
        function rather than inside it.
    """
    check_resampling_arguments(n_resamples, confidence_level, method)

    members, starts, sizes = _cluster_layout(cluster, n_items, unit)
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
            stacklevel=3,
        )

    if isinstance(rng, np.random.Generator):
        generator, seed = rng, None
    else:
        seed = secrets.randbits(64) if rng is None else int(rng)
        generator = np.random.default_rng(seed)

    estimate = float(statistic(np.arange(n_items, dtype=np.intp)))
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = _draw_indices(generator, members, starts, sizes)
        values[i] = statistic(idx)

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
            stacklevel=3,
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
            n_items,
            members,
            starts,
            sizes,
        )

    return ClusterBootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_clusters=n_clusters,
        cluster_sizes=(
            int(sizes.min()),
            int(np.median(sizes)),
            int(sizes.max()),
        ),
        n_valid=n_valid,
        seed=seed,
        distribution=distribution,
    )
