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

How the inversion is done, and how it can fail
----------------------------------------------
Solving for an effect, for clusters, or for items per cluster all reduce to one
one-dimensional root-find, and it is on none of those three. Write ``lam`` for
the effect divided by its own standard error. The rejection probability of the
normal test depends on the design *only* through ``lam``, so inverting the power
function once -- for the ``lam`` at which the test reaches the requested power
-- leaves every remaining step a closed form:

.. code-block:: text

    MDE            effect = lam * sd / sqrt(n_eff)
    clusters       n_eff  = (lam * sd / effect) ** 2,  k = n_eff * DE / m_bar
    items          m_bar  = n_eff * (1 - rho) / (k - n_eff * rho)

The last line is the ceiling of the opening section written out as arithmetic:
its denominator vanishes at ``n_eff = k / rho``, and past that point no item
count solves it at all. That is checked before the division rather than after,
so the impossible request raises with the ceiling's value instead of returning
an enormous number or an infinity.

``lam`` is where :func:`scipy.optimize.brentq` is used, on the bracket
``[0, z_crit + z_power]``. Neither end is searched for; both are known before
the call. At ``lam = 0`` the power is exactly the test's level, which lies below
any target this function accepts -- a target at or below ``alpha`` is refused
earlier, on the ground that an effect of zero already reaches it. The upper end
is the textbook one-sided sample-size solution, which a two-sided test overshoots
by the far-tail term ``Phi(-lam - z_crit)``, a strictly positive quantity, so the
root is bracketed from both sides by construction. For a one-sided alternative
that upper end *is* the root and brentq returns it directly. Only if
floating-point evaluation leaves the upper end a hair on the wrong side is it
doubled, and then at most sixty times.

**Non-convergence is not silent.** brentq is called with ``full_output=True``
and its ``converged`` flag is checked; a bracket that cannot be established, or
a run that exhausts its iterations, raises :class:`RuntimeError` naming the
target power, the level, and the alternative it was solving at. There is no
fallback value, no best-effort return and no warning-and-continue: a power
figure that quietly came back unconverged would be worse than no figure, because
nothing downstream could tell the two apart.
"""

from __future__ import annotations

import math
import secrets
import warnings
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from evalstat.bootstrap import MIN_CLUSTERS, FewClustersWarning, paired_bootstrap

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
        that this function does not make silently. ``power`` is the requested
        target except where solving for ``items_per_cluster`` hit the clamp at
        one item, in which case it is the power the returned design has.
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
    if not np.all(np.isfinite(m)) or np.any(m < 1.0):
        raise ValueError(
            f"items_per_cluster must be finite and at least 1, got "
            f"{items_per_cluster!r}"
        )
    return m, _as_rho(rho)


def _as_rho(rho: float | ArrayLike) -> NDArray[np.float64]:
    """Validate an intra-cluster correlation on its own.

    Split out of :func:`_as_design` because ``power_analysis`` has to check
    ``rho`` even in the call where ``items_per_cluster`` is the argument being
    solved for and there is no cluster size to check it beside.
    """
    r = np.asarray(rho, dtype=np.float64)
    if not np.all(np.isfinite(r)) or np.any(r < 0.0) or np.any(r >= 1.0):
        raise ValueError(
            f"rho must lie in [0, 1), got {rho!r}. At rho = 1 the items of a "
            "cluster are copies of one another and the effective sample is the "
            "cluster count itself -- that is a design with one item per "
            "cluster, and saying so is clearer than deriving it here"
        )
    return r


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


# brentq is given a bracket whose ends are derived rather than searched for, so
# these govern only the final polish; see the module docstring.
_BRENTQ_XTOL = 1e-13
_BRENTQ_RTOL = 4.0 * float(np.finfo(np.float64).eps)
_BRENTQ_MAXITER = 200
_MAX_BRACKET_DOUBLINGS = 60


def _z_crit(alpha: float, alternative: Alternative) -> float:
    """Critical value of the standard normal for a test at this level."""
    tail = alpha / 2.0 if alternative == "two-sided" else alpha
    return float(norm.ppf(1.0 - tail))


def _power_from_lambda(
    lam: NDArray[np.float64] | float,
    z_crit: float,
    alternative: Alternative,
) -> NDArray[np.float64]:
    """Rejection probability at signed noncentrality ``lam``.

    ``lam`` is the effect divided by its own standard error and carries the
    effect's sign, so ``lam = 0`` returns the test's level exactly and an effect
    pointing against a one-sided alternative returns less than it.
    """
    lam_arr = np.asarray(lam, dtype=np.float64)
    if alternative == "greater":
        power = norm.cdf(lam_arr - z_crit)
    elif alternative == "less":
        power = norm.cdf(-lam_arr - z_crit)
    else:
        power = norm.cdf(lam_arr - z_crit) + norm.cdf(-lam_arr - z_crit)
    return np.asarray(power, dtype=np.float64)


def _required_lambda(
    target_power: float, alpha: float, alternative: Alternative
) -> float:
    """Invert the power function: the noncentrality that reaches ``target_power``.

    The only root-find in the module. Bracket, guarantees and failure behaviour
    are described in the module docstring; the short version is that the lower
    end is ``0``, where the power is exactly ``alpha``, and the upper end is the
    one-sided closed form ``z_crit + z_power``, which a two-sided test passes.
    """
    z_crit = _z_crit(alpha, alternative)

    def shortfall(lam: float) -> float:
        return float(_power_from_lambda(lam, z_crit, alternative)) - target_power

    high = z_crit + float(norm.ppf(target_power))
    doublings = 0
    while shortfall(high) < 0.0 and doublings < _MAX_BRACKET_DOUBLINGS:
        # Reached only when the closed-form end lands a rounding error short of
        # the target, which happens for a one-sided alternative where that end
        # is the root itself. The additive term carries a high of exactly zero.
        high = 2.0 * high + 1e-12
        doublings += 1
    if shortfall(high) < 0.0:
        raise RuntimeError(
            f"could not bracket the noncentrality reaching power "
            f"{target_power} at alpha={alpha}, alternative={alternative!r}: "
            f"the power is still below target at lambda={high}. This is a bug "
            "in the bracketing, not a property of the design"
        )

    root, results = brentq(
        shortfall,
        0.0,
        high,
        xtol=_BRENTQ_XTOL,
        rtol=_BRENTQ_RTOL,
        maxiter=_BRENTQ_MAXITER,
        full_output=True,
    )
    if not results.converged:
        raise RuntimeError(
            f"root-finding for power {target_power} at alpha={alpha}, "
            f"alternative={alternative!r} did not converge in "
            f"{_BRENTQ_MAXITER} iterations ({results.flag}); no value is "
            "returned, because an unconverged power figure is not "
            "distinguishable downstream from a converged one"
        )
    return float(root)


def _informative_n(
    n_eff: NDArray[np.float64], statistic: Statistic, tie_rate: float
) -> NDArray[np.float64]:
    """Effective sample carrying information about the statistic.

    Ties are removed from the *effective* sample, not from the nominal item
    count: the clustering discount applies first and the tie discount applies to
    what survives it. Applying ``(1 - tie_rate)`` to the item count instead
    would credit the design with information the clustering already spent.
    """
    if statistic == "mean":
        return n_eff
    return n_eff * (1.0 - tie_rate)


def _lambda_of(
    effect: float,
    n_eff: NDArray[np.float64],
    statistic: Statistic,
    sd: float | None,
    tie_rate: float,
    variance_under: Literal["null", "alternative"],
) -> NDArray[np.float64]:
    """Signed noncentrality of ``effect`` at this design.

    Written as a multiplication by ``sqrt(n)`` rather than a division by a
    standard error so that a zero sample gives a zero noncentrality instead of
    a division by zero.
    """
    n_info = _informative_n(n_eff, statistic, tie_rate)
    if statistic == "mean":
        assert sd is not None  # guaranteed by validation above
        return (effect / sd) * np.sqrt(n_info)
    variance = 0.25 if variance_under == "null" else effect * (1.0 - effect)
    return (effect - 0.5) * np.sqrt(n_info / variance)


def _effect_of(
    lam: float,
    n_eff: NDArray[np.float64],
    statistic: Statistic,
    sd: float | None,
    tie_rate: float,
    variance_under: Literal["null", "alternative"],
    alternative: Alternative,
) -> NDArray[np.float64]:
    """Smallest detectable effect at noncentrality ``lam``.

    Returned on the side the alternative can detect: negative, or below the
    null rate of 0.5, for ``less``; positive, or above 0.5, otherwise.
    """
    n_info = _informative_n(n_eff, statistic, tie_rate)
    sign = -1.0 if alternative == "less" else 1.0
    if statistic == "mean":
        assert sd is not None  # guaranteed by validation above
        return sign * lam * sd / np.sqrt(n_info)
    if variance_under == "null":
        return 0.5 + sign * lam * 0.5 / np.sqrt(n_info)
    # Solving u = lam * sqrt(p (1 - p) / n) with u = p - 0.5, so that
    # p (1 - p) = 0.25 - u**2, gives u**2 (1 + c) = 0.25 c for c = lam**2 / n.
    c = lam**2 / n_info
    return 0.5 + sign * 0.5 * np.sqrt(c / (1.0 + c))


def _required_n_eff(
    lam: float,
    effect: float,
    statistic: Statistic,
    sd: float | None,
    tie_rate: float,
    variance_under: Literal["null", "alternative"],
) -> float:
    """Effective sample at which ``effect`` reaches noncentrality ``lam``.

    The inverse of :func:`_lambda_of` in its sample-size argument. Scalar: it
    depends on the effect and the statistic, and not on how the clustering
    arrived at the effective sample, which is why one value serves a whole
    ``rho`` range.
    """
    if statistic == "mean":
        assert sd is not None  # guaranteed by validation above
        return float((lam * sd / abs(effect)) ** 2)
    variance = 0.25 if variance_under == "null" else effect * (1.0 - effect)
    n_info = lam**2 * variance / (effect - 0.5) ** 2
    return float(n_info / (1.0 - tie_rate))


def _slot_message(slots: dict[str, float | None], missing: list[str]) -> str:
    """Explain a wrong number of ``None`` slots, naming every slot by its state.

    The four-slot signature is the one place this function can be got wrong
    without noticing, so the message counts what was found rather than
    restating the rule and leaving the caller to compare.
    """
    names = list(slots)
    listed = ", ".join(names[:-1]) + f" and {names[-1]}"
    supplied = ", ".join(f"{k}={v!r}" for k, v in slots.items() if v is not None)
    if not missing:
        return (
            f"exactly one of {listed} must be None -- the one passed as None is "
            f"the one solved for, and all four were supplied: {supplied}. Note "
            "that power defaults to 0.80, so a call that never mentions power "
            "still supplies it; pass power=None to solve for power"
        )
    return (
        f"exactly one of {listed} must be None -- the one passed as None is the "
        f"one solved for, and {len(missing)} were None: {', '.join(missing)}. "
        + (f"Supplied: {supplied}. " if supplied else "Nothing else was supplied. ")
        + "Solve for one at a time; to sweep a second quantity, call this "
        "function once per value of it"
    )


# The mean's grid is symmetric and odd-length so that the null sits on a grid
# point exactly rather than near one; the rate's grid is the set of points
# actually simulated, so it is kept short.
_SIM_GRID_POINTS = 41
_RATE_GRID_POINTS = 9
_MAX_BISECTIONS = 200


def _preference_rate(d: NDArray[np.float64]) -> float:
    """Share of non-tied items favouring the first system; NaN when all tied.

    The same definition :func:`power_analysis` documents and
    :mod:`evalstat.bootstrap`'s tests use. Ties leave the denominator; they are
    not counted as half a win, because an abstention is not half a preference.
    """
    decided = int(np.count_nonzero(d != 0.0))
    if decided == 0:
        return float("nan")
    return float(np.count_nonzero(d > 0.0)) / decided


def _cluster_labels(n_clusters: int, items_per_cluster: int) -> NDArray[np.intp]:
    """Contiguous cluster labels, one per item."""
    return np.repeat(np.arange(n_clusters), items_per_cluster).astype(np.intp)


def _draw_mean_differences(
    generator: np.random.Generator,
    n_clusters: int,
    items_per_cluster: int,
    rho: float,
    sd: float,
) -> NDArray[np.float64]:
    """Paired differences at a zero effect, total sd ``sd``, correlation ``rho``.

    ``d_ij = a_i + e_ij`` with ``var(a) = sd**2 * rho`` and
    ``var(e) = sd**2 * (1 - rho)``. This is the generator
    ``tests/test_bootstrap.py`` measures coverage with, at ``mu = 0``, so the
    power computed here and the coverage measured there describe one kind of
    data rather than two.
    """
    between = generator.normal(0.0, sd * math.sqrt(rho), size=n_clusters)
    within = generator.normal(
        0.0, sd * math.sqrt(1.0 - rho), size=(n_clusters, items_per_cluster)
    )
    drawn: NDArray[np.float64] = (between[:, None] + within).ravel()
    return drawn


def _draw_preferences(
    generator: np.random.Generator,
    n_clusters: int,
    items_per_cluster: int,
    rho: float,
    rate: float,
    tie_rate: float,
) -> NDArray[np.float64]:
    """Clustered preferences as -1, 0, +1 with marginal win rate ``rate``.

    A latent normal carries the clustering: ``var(a) = rho`` and
    ``var(e) = 1 - rho`` give it unit variance, so it crosses zero with
    probability ``Phi(mu)`` and ``mu = Phi^-1(rate)`` hits the requested rate
    exactly. Ties are drawn independently at ``tie_rate``, which is assumption 8
    of :func:`power_analysis` and not a claim about raters: real ties cluster,
    and a design whose conclusion turns on that should not read it off this
    generator.
    """
    mu = float(norm.ppf(rate))
    between = generator.normal(0.0, math.sqrt(rho), size=n_clusters)
    within = generator.normal(
        0.0, math.sqrt(1.0 - rho), size=(n_clusters, items_per_cluster)
    )
    decided = np.where(mu + between[:, None] + within > 0.0, 1.0, -1.0)
    if tie_rate > 0.0:
        tied = generator.random((n_clusters, items_per_cluster)) < tie_rate
        decided = np.where(tied, 0.0, decided)
    drawn: NDArray[np.float64] = decided.ravel().astype(np.float64)
    return drawn


def _mean_interval_bounds(
    generator: np.random.Generator,
    n_clusters: int,
    items_per_cluster: int,
    rho: float,
    sd: float,
    n_sim: int,
    n_resamples: int,
    confidence_level: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interval endpoints for ``n_sim`` datasets drawn at a zero effect.

    Only the endpoints are kept because for the mean they are the whole power
    curve. Adding ``delta`` to every difference shifts the estimate and every
    resampled mean by ``delta``, so the interval becomes ``[lo + delta,
    hi + delta]`` exactly, and one run per dataset answers every effect.
    """
    labels = _cluster_labels(n_clusters, items_per_cluster)
    low = np.empty(n_sim, dtype=np.float64)
    high = np.empty(n_sim, dtype=np.float64)
    with warnings.catch_warnings():
        # power_analysis has already warned once for this design; n_sim further
        # copies of the same sentence carry no information the first did not.
        warnings.simplefilter("ignore", FewClustersWarning)
        for i in range(n_sim):
            diff = _draw_mean_differences(
                generator, n_clusters, items_per_cluster, rho, sd
            )
            interval = paired_bootstrap(
                diff,
                cluster=labels,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                rng=generator,
            )
            low[i] = interval.ci_low
            high[i] = interval.ci_high
    return low, high


def _mean_rejection(
    low: NDArray[np.float64], high: NDArray[np.float64], effect: float
) -> float:
    """Share of the simulated intervals that exclude zero at ``effect``."""
    return float(np.mean((low + effect > 0.0) | (high + effect < 0.0)))


def _mean_rejection_curve(
    low: NDArray[np.float64], high: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """:func:`_mean_rejection` over a whole grid, without re-resampling."""
    shifted_low = low[None, :] + grid[:, None]
    shifted_high = high[None, :] + grid[:, None]
    rate: NDArray[np.float64] = np.mean(
        (shifted_low > 0.0) | (shifted_high < 0.0), axis=1, dtype=np.float64
    )
    return rate


def _solve_mean_mde(
    low: NDArray[np.float64], high: NDArray[np.float64], target: float
) -> float:
    """Smallest effect whose empirical rejection rate reaches ``target``.

    Bisection rather than :func:`scipy.optimize.brentq`, and the difference is
    not a preference. The empirical curve is a step function with one jump per
    simulated dataset; brentq interpolates towards a continuous crossing that
    does not exist here, while bisection converges on the jump, which is the
    answer. The bracket is derived rather than searched: at ``-min(lo)`` every
    simulated interval has been shifted clear of zero, so the rate is one there,
    and at zero it is the design's own error rate.

    Raises
    ------
    RuntimeError
        If the bracket cannot be established -- the rate fails to reach the
        target even where every interval has cleared zero. Nothing is returned
        in that case.
    """
    if _mean_rejection(low, high, 0.0) >= target:
        # The design already rejects this often under the null. Reporting a
        # positive MDE would describe detection that is not detection.
        return 0.0
    ceiling = float(-np.min(low)) * (1.0 + 1e-9) + 1e-12
    if ceiling <= 0.0 or _mean_rejection(low, high, ceiling) < target:
        raise RuntimeError(
            f"the simulated rejection rate never reaches {target}: at the "
            f"effect where every simulated interval has cleared zero it is "
            f"{_mean_rejection(low, high, max(ceiling, 0.0))}. This is a bug in "
            "the bracketing rather than a property of the design"
        )
    lower, upper = 0.0, ceiling
    for _ in range(_MAX_BISECTIONS):
        middle = 0.5 * (lower + upper)
        if _mean_rejection(low, high, middle) >= target:
            upper = middle
        else:
            lower = middle
        if upper - lower <= 1e-12 * max(1.0, upper):
            break
    return upper


def _rate_rejection(
    generator: np.random.Generator,
    n_clusters: int,
    items_per_cluster: int,
    rho: float,
    rate: float,
    tie_rate: float,
    n_sim: int,
    n_resamples: int,
    confidence_level: float,
) -> float:
    """Share of simulated intervals excluding the null rate of 0.5.

    A rate is not equivariant under a location shift -- moving it changes its
    variance and its tie structure -- so this is paid for once per grid point
    rather than recovered from a single run.
    """
    labels = _cluster_labels(n_clusters, items_per_cluster)
    rejected = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FewClustersWarning)
        for _ in range(n_sim):
            diff = _draw_preferences(
                generator, n_clusters, items_per_cluster, rho, rate, tie_rate
            )
            interval = paired_bootstrap(
                diff,
                cluster=labels,
                statistic=_preference_rate,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                rng=generator,
            )
            rejected += int(interval.ci_low > 0.5 or interval.ci_high < 0.5)
    return rejected / n_sim


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
        than answered with a large number. At the other end the answer is
        clamped at 1: a design that already reaches the target with one item per
        cluster needs one, and a fractional item count would be arithmetic
        rather than a design. Where that clamp binds, the returned ``power`` is
        the power the clamped design actually has, which is above the one
        requested -- the four quantities always describe a single design.
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
        ``items_per_cluster`` when the target exceeds ``n_clusters / rho``. If
        the target power does not exceed the level the test runs at, which an
        effect of zero already reaches.
    RuntimeError
        If the root-find behind the inversion fails to converge or cannot be
        bracketed. Nothing is returned in that case and no warning is issued in
        place of the value; see the module docstring for the bracket and why
        the failure is loud.

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
    # The four-slot signature is checked before anything else: it is the one
    # way to call this function wrongly that a reader cannot see in the call.
    slots: dict[str, float | None] = {
        "effect": effect,
        "n_clusters": n_clusters,
        "items_per_cluster": items_per_cluster,
        "power": power,
    }
    missing = [name for name, value in slots.items() if value is None]
    if len(missing) != 1:
        raise ValueError(_slot_message(slots, missing))
    solved_for = cast(Solvable, missing[0])

    if statistic not in ("mean", "preference_rate"):
        raise ValueError(f"unknown statistic {statistic!r}")
    if method not in ("simulation", "analytic"):
        raise ValueError(f"unknown method {method!r}")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"unknown alternative {alternative!r}")
    if variance_under not in ("null", "alternative"):
        raise ValueError(f"unknown variance_under {variance_under!r}")

    rho_arr = np.atleast_1d(_as_rho(rho))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError(f"power must lie in (0, 1), got {power!r}")

    if effect is not None:
        if statistic == "mean":
            if effect == 0.0 or not np.isfinite(effect):
                raise ValueError(
                    f"effect must be finite and non-zero, got {effect!r}. No "
                    "sample size detects a difference of exactly zero: the "
                    "power of the test against it is the level alpha itself"
                )
        elif not 0.0 < effect < 1.0 or effect == 0.5:
            raise ValueError(
                f"for preference_rate, effect is the rate itself and must lie "
                f"in (0, 1) away from the null of 0.5, got {effect!r}"
            )

    if statistic == "mean":
        if sd is None:
            raise ValueError(
                "sd is required for statistic='mean' and has no default: the "
                "effect is measured in units of sd, so an assumed sd would "
                "silently rescale the answer. Pass sd=1.0 to work in "
                "standardised units"
            )
        if not np.isfinite(sd) or sd <= 0.0:
            raise ValueError(f"sd must be finite and positive, got {sd!r}")
        if tie_rate != 0.0:
            raise ValueError(
                f"tie_rate applies to preference_rate only, got {tie_rate!r} "
                "for statistic='mean'. A mean has no ties to discard: a paired "
                "difference of exactly zero is an observation, not an "
                "abstention"
            )
    else:
        if sd is not None:
            raise ValueError(
                f"sd applies to statistic='mean' only, got {sd!r} for "
                "preference_rate, whose variance comes from the rate itself"
            )
        if not 0.0 <= tie_rate < 1.0:
            raise ValueError(
                f"tie_rate must lie in [0, 1), got {tie_rate!r}; at a tie rate "
                "of 1 no item carries a preference and there is no design to "
                "compute power for"
            )

    if coverage is not None:
        if method == "simulation":
            raise ValueError(
                "coverage is for the analytic route only. The simulation route "
                "measures the interval procedure's real coverage at this "
                "design and reports it as coverage_measured; a supplied figure "
                "would be a second and contradictory claim about the same "
                "quantity"
            )
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must lie in (0, 1), got {coverage!r}")

    if n_clusters is not None and (not np.isfinite(n_clusters) or n_clusters < 2):
        raise ValueError(
            f"n_clusters must be finite and at least 2, got {n_clusters!r}; "
            "a single cluster carries no information about variation between "
            "clusters, which is the variation this correction is about"
        )
    if items_per_cluster is not None:
        _as_design(items_per_cluster, rho_arr)

    ones = np.ones_like(rho_arr)

    if method == "simulation":
        if alternative != "two-sided":
            raise NotImplementedError(
                f"the simulation route runs the two-sided interval rule only, "
                f"got alternative={alternative!r}. paired_bootstrap returns a "
                "two-sided interval, and turning it into a one-sided test at "
                "this alpha means deciding which interval to build it from -- a "
                "choice about the test, not about this code, and one nothing "
                "here has tested. Use method='analytic' for a one-sided figure"
            )
        if solved_for in ("n_clusters", "items_per_cluster"):
            raise NotImplementedError(
                f"the simulation route solves for effect or power, not for "
                f"{solved_for}. Solving for a design quantity means simulating "
                "at each candidate design and searching over them, so the cost "
                "is one full simulation per step rather than one in total, and "
                "no shortcut like the mean's location-equivariance applies. Use "
                "method='analytic' to size the design, then check the design it "
                "returns with a simulation run"
            )
        assert n_clusters is not None and items_per_cluster is not None
        if float(items_per_cluster) != int(items_per_cluster):
            raise ValueError(
                f"the simulation route needs a whole number of items per "
                f"cluster to generate, got {items_per_cluster!r}. m_bar is a "
                "variance-weighted mean and may be fractional on the analytic "
                "route, where it enters only through the design effect; here it "
                "is a count of items to draw"
            )

        if isinstance(rng, np.random.Generator):
            generator, seed = rng, None
        else:
            seed = secrets.randbits(64) if rng is None else int(rng)
            generator = np.random.default_rng(seed)

        clusters = int(n_clusters)
        items = int(items_per_cluster)
        confidence_level = 1.0 - alpha
        m_arr = ones * float(items)
        k_arr = ones * float(clusters)
        de = 1.0 + (m_arr - 1.0) * rho_arr
        n_eff = k_arr * m_arr / de

        if clusters < MIN_CLUSTERS:
            warnings.warn(
                f"{clusters} clusters is below {MIN_CLUSTERS}; the cluster "
                "bootstrap being simulated under-covers at that count, so it "
                "rejects more often than alpha. The simulation measures that "
                "rather than assuming it away -- read coverage_measured beside "
                "the power",
                FewClustersWarning,
                stacklevel=2,
            )

        power_arr = np.empty_like(rho_arr)
        effect_arr = np.empty_like(rho_arr)
        measured = np.empty_like(rho_arr)

        if statistic == "mean":
            assert sd is not None  # guaranteed by validation above
            bounds = [
                _mean_interval_bounds(
                    generator,
                    clusters,
                    items,
                    float(r),
                    sd,
                    n_sim,
                    n_resamples,
                    confidence_level,
                )
                for r in rho_arr
            ]
            for j, (low, high) in enumerate(bounds):
                # The zero point of the same run: the rejection rate under the
                # null is this design's real error rate, so its complement is
                # the coverage. Measured at the design being asked about, not
                # carried to it from another one.
                measured[j] = 1.0 - _mean_rejection(low, high, 0.0)
                if solved_for == "effect":
                    assert power is not None  # guaranteed by the slot check
                    effect_arr[j] = _solve_mean_mde(low, high, power)
                    power_arr[j] = power
                else:
                    assert effect is not None  # guaranteed by the slot check
                    effect_arr[j] = effect
                    power_arr[j] = _mean_rejection(low, high, effect)

            span = float(np.max(np.abs(effect_arr))) * 1.5
            grid = np.linspace(-span, span, _SIM_GRID_POINTS)
            grid[_SIM_GRID_POINTS // 2] = 0.0  # the null, exactly on a point
            curve_power = np.vstack(
                [_mean_rejection_curve(low, high, grid) for low, high in bounds]
            )
        else:
            if solved_for == "effect":
                assert power is not None  # guaranteed by the slot check
                # For a rate the grid is where the simulations happen, so it is
                # fixed before any of them run. The analytic MDE sets the scale
                # and is known to be optimistic, hence the headroom above it.
                lam_req = _required_lambda(power, alpha, alternative)
                analytic = _effect_of(
                    lam_req,
                    n_eff,
                    statistic,
                    sd,
                    tie_rate,
                    variance_under,
                    alternative,
                )
                top = min(0.5 + 1.8 * (float(np.max(analytic)) - 0.5), 0.999)
                grid = np.linspace(0.5, top, _RATE_GRID_POINTS)
            else:
                assert effect is not None  # guaranteed by the slot check
                # Two points: the null, which measures coverage, and the rate
                # asked about. A denser grid would be simulation nobody asked
                # for -- the solution is read off one point, not a curve.
                grid = np.array([0.5, float(effect)], dtype=np.float64)

            curve_power = np.vstack(
                [
                    [
                        _rate_rejection(
                            generator,
                            clusters,
                            items,
                            float(r),
                            float(point),
                            tie_rate,
                            n_sim,
                            n_resamples,
                            confidence_level,
                        )
                        for point in grid
                    ]
                    for r in rho_arr
                ]
            )
            for j in range(rho_arr.size):
                measured[j] = 1.0 - curve_power[j, 0]
                if solved_for == "effect":
                    assert power is not None  # guaranteed by the slot check
                    # Monotone in the rate up to Monte Carlo noise; the running
                    # maximum imposes that so the read-off cannot go backwards
                    # on a wiggle, which np.interp would otherwise let it do.
                    rising = np.maximum.accumulate(curve_power[j])
                    if rising[-1] < power:
                        raise RuntimeError(
                            f"the simulated rejection rate reaches only "
                            f"{rising[-1]} at a rate of {grid[-1]}, short of "
                            f"the requested {power}; the grid this MDE would be "
                            "read off does not contain the answer"
                        )
                    effect_arr[j] = float(np.interp(power, rising, grid))
                    power_arr[j] = power
                else:
                    assert effect is not None  # guaranteed by the slot check
                    effect_arr[j] = float(effect)
                    power_arr[j] = curve_power[j, -1]

        mc_se = np.sqrt(power_arr * (1.0 - power_arr) / n_sim)
        with np.errstate(divide="ignore"):
            ceiling = np.where(rho_arr > 0.0, k_arr / rho_arr, np.inf)
        return PowerAnalysisResult(
            solved_for=solved_for,
            statistic=statistic,
            effect=effect_arr,
            power=power_arr,
            n_clusters=k_arr,
            items_per_cluster=m_arr,
            n_items=k_arr * m_arr,
            n_eff=n_eff,
            design_effect=de,
            n_eff_ceiling=ceiling,
            rho=rho_arr,
            sd=sd,
            tie_rate=tie_rate,
            alpha=alpha,
            alternative=alternative,
            variance_under=variance_under,
            coverage=None,
            coverage_measured=measured,
            coverage_source="measured",
            method=method,
            n_sim=n_sim,
            n_resamples=n_resamples,
            mc_se=mc_se,
            seed=seed,
            curve=PowerCurve(
                effect=grid,
                power=curve_power,
                mc_se=np.sqrt(curve_power * (1.0 - curve_power) / n_sim),
            ),
        )

    if statistic == "preference_rate":
        warnings.warn(
            "the analytic route treats the non-tied count as a fixed, known "
            "quantity, when ties are counted from the same clustered data as "
            "the preferences; the variance it uses is too small, so this power "
            "is too high and this MDE too small",
            AnalyticProportionWarning,
            stacklevel=2,
        )

    # A supplied coverage replaces the nominal level outright rather than
    # correcting it: 1 - coverage is the rate at which the procedure actually
    # rejects, and that is the level the calculation should run at.
    alpha_eff = alpha if coverage is None else 1.0 - coverage
    coverage_source: Literal["nominal", "supplied", "measured"] = (
        "nominal" if coverage is None else "supplied"
    )
    z_crit = _z_crit(alpha_eff, alternative)

    if solved_for == "power":
        assert effect is not None  # guaranteed by the slot check
        assert n_clusters is not None and items_per_cluster is not None
        m_arr = ones * float(items_per_cluster)
        k_arr = ones * float(n_clusters)
        de = 1.0 + (m_arr - 1.0) * rho_arr
        n_eff = k_arr * m_arr / de
        lam = _lambda_of(effect, n_eff, statistic, sd, tie_rate, variance_under)
        power_arr = _power_from_lambda(lam, z_crit, alternative)
        effect_arr = ones * float(effect)
    else:
        assert power is not None  # guaranteed by the slot check
        if power <= alpha_eff:
            raise ValueError(
                f"power must exceed the level the test runs at, {alpha_eff}, "
                f"got {power}: an effect of zero is already declared "
                "significant that often, so there is nothing to solve for"
            )
        lam_req = _required_lambda(power, alpha_eff, alternative)
        power_arr = ones * float(power)

        if solved_for == "effect":
            assert n_clusters is not None and items_per_cluster is not None
            m_arr = ones * float(items_per_cluster)
            k_arr = ones * float(n_clusters)
            de = 1.0 + (m_arr - 1.0) * rho_arr
            n_eff = k_arr * m_arr / de
            effect_arr = _effect_of(
                lam_req, n_eff, statistic, sd, tie_rate, variance_under, alternative
            )
        else:
            assert effect is not None  # guaranteed by the slot check
            effect_arr = ones * float(effect)
            n_eff_req = _required_n_eff(
                lam_req, effect, statistic, sd, tie_rate, variance_under
            )
            if solved_for == "n_clusters":
                assert items_per_cluster is not None
                m_arr = ones * float(items_per_cluster)
                de = 1.0 + (m_arr - 1.0) * rho_arr
                # k enters n_eff linearly at fixed m_bar, so no search is
                # needed once the required effective sample is known.
                k_arr = n_eff_req * de / m_arr
            else:
                assert n_clusters is not None
                k_arr = ones * float(n_clusters)
                with np.errstate(divide="ignore"):
                    reachable = np.where(rho_arr > 0.0, k_arr / rho_arr, np.inf)
                blocked = n_eff_req >= reachable
                if np.any(blocked):
                    i = int(np.argmax(blocked))
                    raise ValueError(
                        f"no items_per_cluster reaches power {power} at "
                        f"n_clusters={n_clusters}: with rho = {rho_arr[i]:g} "
                        f"the effective sample is bounded above by "
                        f"n_clusters / rho = {reachable[i]:g}, and this target "
                        f"needs {n_eff_req:g}. Effective sample size is bought "
                        "with clusters; items past the first few in a cluster "
                        "buy rating work"
                    )
                # Inverting n_eff = k * m / (1 + (m - 1) * rho) for m. The
                # ceiling above is this denominator reaching zero. Clamped at
                # one item per cluster: where the design already reaches the
                # target with a single item, one is the answer, and a fractional
                # item count would be arithmetic rather than a design.
                unclamped = n_eff_req * (1.0 - rho_arr) / (k_arr - n_eff_req * rho_arr)
                clamped = unclamped < 1.0
                m_arr = np.maximum(unclamped, 1.0)
                de = 1.0 + (m_arr - 1.0) * rho_arr
            n_eff = k_arr * m_arr / de
            if solved_for == "items_per_cluster" and np.any(clamped):
                # The clamp raises the design above the power that was asked
                # for, so the requested figure is no longer this design's. The
                # four quantities have to describe one design between them, not
                # three of it and the target of the fourth.
                achieved = _power_from_lambda(
                    _lambda_of(effect, n_eff, statistic, sd, tie_rate, variance_under),
                    z_crit,
                    alternative,
                )
                power_arr = np.where(clamped, achieved, power_arr)

    fewest = float(np.min(k_arr))
    if fewest < MIN_CLUSTERS:
        warnings.warn(
            f"{fewest:g} clusters is below {MIN_CLUSTERS}; the cluster "
            "bootstrap this power figure describes under-covers at that count, "
            "so it rejects more often than alpha — the power quoted here "
            "belongs to a test running at an error rate above the one stated",
            FewClustersWarning,
            stacklevel=2,
        )

    with np.errstate(divide="ignore"):
        ceiling = np.where(rho_arr > 0.0, k_arr / rho_arr, np.inf)

    return PowerAnalysisResult(
        solved_for=solved_for,
        statistic=statistic,
        effect=effect_arr,
        power=power_arr,
        n_clusters=k_arr,
        items_per_cluster=m_arr,
        n_items=k_arr * m_arr,
        n_eff=n_eff,
        design_effect=de,
        n_eff_ceiling=ceiling,
        rho=rho_arr,
        sd=sd,
        tie_rate=tie_rate,
        alpha=alpha,
        alternative=alternative,
        variance_under=variance_under,
        coverage=coverage,
        coverage_measured=None,
        coverage_source=coverage_source,
        method=method,
        n_sim=None,
        n_resamples=None,
        mc_se=None,
        seed=None,
        curve=None,
    )
