"""Power and minimum detectable effect for clustered paired evaluations.

Two questions, one function. Forwards: how many clusters does it take to detect
an effect of a stated size? Inverted: with the design already fixed by
collection and rating capacity, what is the smallest effect that design can
detect? The second question is the one an evaluation usually faces, and it is
the one most tooling answers badly -- not because the arithmetic is hard, but
because the textbook formula is written for the first.

Clustering is the reason this module exists, for the same reason
:mod:`evalstat.bootstrap` exists. Items drawn three to an article, or three to a
recording, carry a shared component, and a power calculation that counts them as
independent promises a sample size that does not exist. The correction runs
through the design effect on an effective sample size

.. code-block:: text

    DE    = 1 + (m_bar - 1) * rho
    n_eff = k * m_bar / DE          -> k / rho as m_bar grows without bound

and the ceiling in that second line is not a footnote. At k = 40 clusters and
rho = 0.2 no number of items per cluster reaches an effective 201: raising items
per cluster from 3 to 15 multiplies the rating load fivefold and moves the
effective sample from 86 to 158, and it can never reach 200. Clusters buy
information; items past the first few mostly buy rating work. When the target is
above the ceiling, :func:`power_analysis` says so with the ceiling's value
rather than returning a very large number of items.

``m_bar`` is the variance-weighted mean cluster size ``sum(m**2) / sum(m)``, not
the plain mean. The two agree when clusters are equal-sized and diverge when
they are not, and the plain mean understates the design effect -- see
:func:`design_effect`.

Why the default is simulation
-----------------------------
The analytic route evaluates a normal approximation at the nominal ``alpha``.
That is fast, familiar, and it assumes the thing :mod:`evalstat.bootstrap`
measured to be false: that the interval procedure runs at its nominal level. At
k = 40, m = 3, rho = 0.2 the nominal 95% cluster bootstrap interval covers about
92.7%. An MDE reported against the nominal level would state a detection
threshold for a 5% test the study will not be running, and put the discrepancy
in a limitations paragraph.

So the default route simulates: draw clustered data under the assumed design,
run :func:`evalstat.paired_bootstrap` on it, and count how often the interval
excludes zero. Nothing is corrected, because nothing is assumed -- the shortfall
is inside the measurement, at the design point being asked about, rather than
extrapolated to it from one previously measured point. The analytic route stays
available as a fast comparison and as the number other tools would report.

Read the two together in the direction they actually run. Under-coverage makes
intervals narrower, so the procedure rejects *more* often than nominal, under
the null as well as under the alternative. The optimism is therefore not in the
power figure but in the claimed error rate: buying a genuine 5% would mean a
wider interval, less power at the same n, and a larger MDE. See
``docs/design/power_analysis.md``.

What simulation costs, and why less than it looks
-------------------------------------------------
For the mean the cost mostly disappears. The mean is equivariant under a
location shift, so adding ``delta`` to every difference shifts the bootstrap
distribution and its interval by ``delta``, and the rejection rule
``0 not in [lo + delta, hi + delta]`` is the same as ``-delta not in [lo, hi]``.
One bootstrap run per simulated dataset therefore yields the whole power curve
over the effect grid exactly, and the curve comes out smooth and monotone, which
is what makes root-finding on it stable. The zero point of that same curve is
the procedure's empirical error rate, so the design's coverage is measured for
free and reported on the result.

A preference rate is not location-equivariant and gets no such discount: its
grid is evaluated point by point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AnalyticProportionWarning",
    "PowerAnalysisResult",
    "PowerCurve",
    "design_effect",
    "effective_n",
    "power_analysis",
]

Statistic = Literal["mean", "preference_rate"]
Alternative = Literal["two-sided", "less", "greater"]
Method = Literal["simulation", "analytic"]
Solvable = Literal["effect", "n_clusters", "items_per_cluster", "power"]


class AnalyticProportionWarning(UserWarning):
    """The analytic route was used for a preference rate, and it runs optimistic.

    The closed form treats the number of non-tied items as a fixed, known
    ``n * (1 - tie_rate)``. It is neither: ties are counted from the same
    clustered data as the preferences, so the denominator is random and its
    variation is itself correlated within clusters. Both omissions push the same
    way. The variance used is too small, so the reported power is too high and
    the reported MDE too small -- the error runs toward believing the design can
    detect a smaller difference than it can.

    The route is kept because it is the number most other tools report and a
    comparison against it is worth having. It is not the route to report from.
    """


@dataclass(frozen=True)
class PowerCurve:
    """Rejection rate over an effect grid, at one design.

    Attributes
    ----------
    effect
        Effect grid, shape ``(n_grid,)``, in the units of the statistic: the
        units of ``sd`` for ``mean``, and a rate in ``[0, 1]`` for
        ``preference_rate``. The grid spans zero, whose rejection rate is the
        procedure's empirical error rate at this design.
    power
        Rejection rate, shape ``(n_rho, n_grid)``.
    mc_se
        Monte Carlo standard error of each entry of ``power``, same shape.
        Differences smaller than a couple of these are noise, and reading a
        curve without them invites reading its wiggles.
    """

    effect: NDArray[np.float64]
    power: NDArray[np.float64]
    mc_se: NDArray[np.float64]


@dataclass(frozen=True)
class PowerAnalysisResult:
    """One power or MDE solution, with the record needed to reproduce it.

    Every array field has shape ``(n_rho,)``, including when ``rho`` was passed
    as a scalar, in which case the length is one. The shape does not depend on
    the shape of the input, so downstream code has one case to handle rather
    than two.

    Attributes
    ----------
    solved_for
        Which argument was passed as ``None`` and solved for.
    statistic
        ``mean`` or ``preference_rate``.
    effect, power, n_clusters, items_per_cluster
        The four solvable quantities, three as supplied and one as solved.
        ``n_clusters`` is **not rounded**: it is the real-valued solution, and
        rounding it is a decision about which side of the target to land on
        that this function does not make silently.
    n_items
        ``n_clusters * items_per_cluster``, the rating load.
    n_eff
        Effective sample size after the design effect.
    design_effect
        ``1 + (items_per_cluster - 1) * rho``.
    n_eff_ceiling
        ``n_clusters / rho``, the effective sample no number of items per
        cluster can pass. Infinite at ``rho = 0``.
    rho
        Intra-cluster correlation the solution was computed at. An array here is
        a sensitivity range, not an estimate.
    sd
        Standard deviation of the paired differences; ``None`` for
        ``preference_rate``.
    tie_rate
        Share of items expected to be ties; zero for ``mean``.
    alpha, alternative, variance_under
        Test as specified.
    coverage
        Real coverage supplied by the caller for the analytic route, or
        ``None``.
    coverage_measured
        Empirical coverage of the interval procedure at this design, read off
        the zero point of the power curve. Populated by the simulation route
        only; it is a measurement of this design, not a figure carried in from
        another one.
    coverage_source
        ``nominal``, ``supplied`` or ``measured``: which of the two fields above
        the reported numbers actually rest on.
    method
        ``simulation`` or ``analytic``.
    n_sim, n_resamples, mc_se, seed
        Simulation provenance; ``None`` on the analytic route. ``seed`` is
        populated even when the caller passed nothing, and is ``None`` only when
        the caller supplied its own generator, whose state cannot be recorded.
    curve
        The power curve the solution was read off, or ``None`` on the analytic
        route, where a closed form can be evaluated anywhere and a grid would be
        an artefact of the implementation rather than a result.
    """

    solved_for: Solvable
    statistic: Statistic
    effect: NDArray[np.float64]
    power: NDArray[np.float64]
    n_clusters: NDArray[np.float64]
    items_per_cluster: NDArray[np.float64]
    n_items: NDArray[np.float64]
    n_eff: NDArray[np.float64]
    design_effect: NDArray[np.float64]
    n_eff_ceiling: NDArray[np.float64]
    rho: NDArray[np.float64]
    sd: float | None
    tie_rate: float
    alpha: float
    alternative: Alternative
    variance_under: Literal["null", "alternative"]
    coverage: float | None
    coverage_measured: NDArray[np.float64] | None
    coverage_source: Literal["nominal", "supplied", "measured"]
    method: Method
    n_sim: int | None
    n_resamples: int | None
    mc_se: NDArray[np.float64] | None
    seed: int | None
    curve: PowerCurve | None


def _as_design(
    items_per_cluster: float | ArrayLike,
    rho: float | ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a cluster size and an intra-cluster correlation together.

    Shared by both helpers so that the two cannot drift into disagreeing about
    what they accept.
    """
    m = np.asarray(items_per_cluster, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.any(m < 1.0):
        raise ValueError(
            f"items_per_cluster must be finite and at least 1, got "
            f"{items_per_cluster!r}"
        )
    if not np.all(np.isfinite(r)) or np.any(r < 0.0) or np.any(r >= 1.0):
        raise ValueError(
            f"rho must lie in [0, 1), got {rho!r}. At rho = 1 the items of a "
            "cluster are copies of one another and the effective sample is the "
            "cluster count itself -- that is a design with one item per "
            "cluster, and saying so is clearer than deriving it here"
        )
    return m, r


def design_effect(
    items_per_cluster: float | ArrayLike,
    rho: float | ArrayLike,
) -> NDArray[np.float64]:
    """Return ``1 + (m_bar - 1) * rho``, the variance inflation from clustering.

    Parameters
    ----------
    items_per_cluster
        ``m_bar``, the variance-weighted mean cluster size. With observed
        cluster sizes ``m`` that is ``(m**2).sum() / m.sum()``, not
        ``m.mean()``. The two agree only when clusters are equal-sized; where
        they differ the plain mean is the smaller, and using it understates the
        design effect, which is the direction that flatters the design. Need not
        be an integer.
    rho
        Intra-cluster correlation of the paired differences, in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        Broadcast over both arguments. Exactly ``1.0`` at ``rho = 0``, so the
        unclustered case passes through this function unchanged rather than
        around it.

    Raises
    ------
    ValueError
        If ``items_per_cluster`` is below 1, or ``rho`` is outside ``[0, 1)``.
    """
    m, r = _as_design(items_per_cluster, rho)
    # Written so that rho = 0 and m = 1 both give exactly 1.0 rather than
    # something that rounds to it: (m - 1) * 0 and 0 * r are both exact.
    inflation: NDArray[np.float64] = 1.0 + (m - 1.0) * r
    return inflation


def effective_n(
    n_clusters: float | ArrayLike,
    items_per_cluster: float | ArrayLike,
    rho: float | ArrayLike,
) -> NDArray[np.float64]:
    """Return ``k * m_bar / design_effect``, the sample size that is really there.

    This is the number to write in a plan, and the number a plan should be
    checked against: with 40 clusters of 3 it is 100 at ``rho = 0.1`` and about
    85.7 at ``rho = 0.2``, against a nominal 120.

    Parameters
    ----------
    n_clusters
        Number of independent clusters, ``k``.
    items_per_cluster
        ``m_bar``; see :func:`design_effect`.
    rho
        Intra-cluster correlation, in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        Broadcast over the arguments. Bounded above by ``n_clusters / rho`` for
        every ``items_per_cluster``.

    Raises
    ------
    ValueError
        If ``n_clusters`` is below 2, ``items_per_cluster`` is below 1, or
        ``rho`` is outside ``[0, 1)``.
    """
    m, r = _as_design(items_per_cluster, rho)
    k = np.asarray(n_clusters, dtype=np.float64)
    if not np.all(np.isfinite(k)) or np.any(k < 2.0):
        raise ValueError(
            f"n_clusters must be finite and at least 2, got {n_clusters!r}; "
            "a single cluster carries no information about variation between "
            "clusters, which is the variation this correction is about"
        )
    n_eff: NDArray[np.float64] = k * m / (1.0 + (m - 1.0) * r)
    return n_eff


def power_analysis(
    *,
    statistic: Statistic = "mean",
    effect: float | None = None,
    n_clusters: int | None = None,
    items_per_cluster: float | None = None,
    power: float | None = 0.80,
    rho: float | ArrayLike,
    sd: float | None = None,
    tie_rate: float = 0.0,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    variance_under: Literal["null", "alternative"] = "null",
    coverage: float | None = None,
    method: Method = "simulation",
    n_sim: int = 2_000,
    n_resamples: int = 999,
    rng: int | np.random.Generator | None = None,
) -> PowerAnalysisResult:
    """Solve one of effect, clusters, items or power from the other three.

    Pass exactly one of ``effect``, ``n_clusters``, ``items_per_cluster`` and
    ``power`` as ``None`` and it is the one solved for. ``effect=None`` is the
    minimum detectable effect; ``n_clusters=None`` is the sample size question.

    **The MDE returned is the effect detectable with probability ``power``**,
    which is what ``power`` is for. It is *not* the expected half-width of the
    confidence interval. Those two are routinely confused and they are not the
    same quantity: an effect equal to the expected half-width is detected about
    half the time, so quoting a half-width as an MDE describes a design at
    roughly 50% power as though it were at 80%. This function does not compute
    the half-width; :func:`evalstat.paired_bootstrap` returns intervals whose
    width can be read directly if that is what is wanted.

    Parameters
    ----------
    statistic
        ``mean`` for a mean paired difference, in the units of ``sd``.
        ``preference_rate`` for the share of non-tied items favouring the first
        system, whose null is 0.5. The two do not come from one formula: the
        variance of a rate moves with the rate, ties make its denominator
        random, and the intra-cluster correlation of a preference decision is a
        different quantity from that of a continuous difference, so ``rho`` for
        one is not ``rho`` for the other.
    effect
        Effect to detect, in the units of the statistic. For
        ``preference_rate`` this is the rate itself, not its distance from 0.5.
        Sign follows :func:`evalstat.paired_bootstrap`: positive favours the
        first system.
    n_clusters
        Number of independent clusters, ``k``. Solving for this returns a
        real-valued answer that the caller rounds; see
        :class:`PowerAnalysisResult`.
    items_per_cluster
        ``m_bar``; see :func:`design_effect`. Solving for this is the request
        that can be impossible: past ``n_clusters / rho`` no item count reaches
        the target, and that is raised as an error naming the ceiling rather
        than answered with a large number.
    power
        Target rejection probability. Defaults to 0.80, which is a convention
        and not a finding; it is the threshold at which an MDE curve is read,
        and a study that wants a different one should say so.
    rho
        Intra-cluster correlation, required, in ``[0, 1)``. There is no default
        because the only available default would be zero, and silently assuming
        independence is the error this package exists to correct. A scalar gives
        one solution; a sequence gives a sensitivity range across it, which is
        the honest form before a pilot has estimated anything.
    sd
        Standard deviation of the paired differences. Required for ``mean`` and
        rejected for ``preference_rate``. There is no default: pass ``1.0``
        deliberately to work in standardised units.
    tie_rate
        Expected share of items on which the rater is indifferent, for
        ``preference_rate``. Ties carry no information, so the information
        available is the non-tied share of an already clustered effective
        sample, not of the nominal item count. Rejected for ``mean``.
    alpha
        Nominal test level. Pass an already-corrected level when the comparison
        sits inside a family; no multiplicity correction is applied here.
    alternative
        ``two-sided`` (default), ``less`` or ``greater``, following
        :mod:`scipy.stats`. A one-sided test buys power by declaring the
        opposite direction unmeasurable, which is a design commitment and not a
        convenience.
    variance_under
        For ``preference_rate``, whether the variance is evaluated at the null
        rate of 0.5 (default, conservative) or at the alternative.
    coverage
        Real coverage of the interval procedure, for the analytic route: supply
        the measured figure and the calculation is done at ``1 - coverage``
        instead of at ``alpha``. Rejected on the simulation route, which
        measures the same quantity itself and where a supplied figure would be a
        second, contradictory claim about it.
    method
        ``simulation`` (default) runs :func:`evalstat.paired_bootstrap` on
        generated data and counts rejections, so the interval procedure's real
        error rate is inside the answer. ``analytic`` evaluates a normal
        approximation at the nominal level; it is fast, it is what other tools
        report, and it assumes the level the bootstrap was measured not to
        deliver.
    n_sim, n_resamples
        Simulated datasets, and resamples per dataset. Simulation route only.
    rng
        Seed, generator, or ``None``. When ``None`` a seed is drawn and recorded
        on the result.

    Returns
    -------
    PowerAnalysisResult
        The solution, the design it implies, and its provenance.

    Raises
    ------
    ValueError
        If other than exactly one of ``effect``, ``n_clusters``,
        ``items_per_cluster`` and ``power`` is ``None`` -- the message names all
        four and says which were supplied. If ``rho`` is outside ``[0, 1)``,
        ``alpha`` or ``power`` outside ``(0, 1)``, ``tie_rate`` outside
        ``[0, 1)``, or ``effect`` is zero. If ``sd`` is missing for ``mean`` or
        supplied for ``preference_rate``, or ``tie_rate`` supplied for ``mean``.
        If ``coverage`` is supplied on the simulation route. If solving for
        ``items_per_cluster`` when the target exceeds ``n_clusters / rho``.

    Warns
    -----
    AnalyticProportionWarning
        The analytic route was used for ``preference_rate``.
    evalstat.FewClustersWarning
        Fewer clusters than :data:`evalstat.MIN_CLUSTERS`. A power figure for a
        design whose intervals are known not to cover is a figure about a
        procedure that is not delivering its stated level.

    Notes
    -----
    Assumptions, and where each one fails:

    1. **Clusters share one exchangeable random component with a single rho.**
       Clusters that differ in difficulty, or a mix of domains, make rho
       heterogeneous, and one number then describes none of them.
    2. **Cluster size enters only through m_bar.** Correct for equal-sized
       clusters; with unequal sizes the variance-weighted mean is the right
       summary and the plain mean understates the design effect.
    3. **The sampling distribution of the statistic is close to normal.** Weak
       for a rate near 0 or 1, and weak for differences that are zero-inflated,
       which paired system outputs often are: two systems that produce the same
       output give a difference of exactly zero.
    4. **sd, or the rate and tie rate, are known constants.** They come from a
       pilot, which estimates them with error. Power depends on the square of
       sd, so a point MDE claims a precision the pilot cannot support; pass a
       range.
    5. **rho is a known constant.** Estimated from few clusters it is estimated
       badly, which is the reason this function accepts a sequence.
    6. **The effect is one constant shift, the same in every cluster.** If a
       system helps mainly on the hard clusters, the unmodelled heterogeneity is
       extra variance and the real power is below the computed power.
    7. **The test runs at the level it claims.** Measured false at k = 40; the
       simulation route is the response, the analytic route inherits the
       assumption.
    8. **Ties are uninformative and their rate is a constant.** Ties cluster,
       so the non-tied count is random and correlated within clusters, and the
       tie rate may itself depend on the difference between systems.
    9. **One comparison.** Nothing here corrects for a family of them.
    10. **"The interval excludes zero" is the decision rule.** The power is the
        power of that rule, and matches a reported test only if the study
        reports the same rule at the same level.
    11. **The analytic route treats variance as free of the effect.** True for
        a mean, false for a rate; hence ``variance_under``.
    12. **This is a design calculation, computed before data.** Recomputing it
        with the observed effect after the fact rewrites the p-value and adds
        nothing; no argument here prevents that, and it remains wrong.

    An open question is recorded in ``docs/design/power_analysis.md``: the
    figure returned is the procedure's power at the error rate the procedure
    actually runs at, which the result reports alongside it. Calibrating the
    interval's level so that the real error rate equals ``alpha``, and reading
    the MDE off that, answers a different question and is not implemented.
    """
    raise NotImplementedError
