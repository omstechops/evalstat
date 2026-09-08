"""Tests for :func:`evalstat.power.power_analysis` and its two helpers.

Written before the implementation, so this file is the specification the body
has to satisfy rather than a description of what it happened to do.

Each case pins the function against an answer known independently of it: the
classical sample size formula with its normal quantiles written out as literals,
the design effect's fixed points and its ceiling, the exact binomial, and
:func:`evalstat.paired_bootstrap` itself. Two are marked slow.

``power_analysis`` is imported from ``evalstat.power`` and not from ``evalstat``
because it is not exported until it works.
"""

import math
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import binom

from evalstat import MIN_CLUSTERS, FewClustersWarning, paired_bootstrap
from evalstat.power import (
    AnalyticProportionWarning,
    design_effect,
    effective_n,
    power_analysis,
)
from test_bootstrap import clustered_differences, preference_rate

# Standard normal quantiles, written out rather than called for, so that the
# reference values below do not come from the same library as the answer.
Z_0975 = 1.959963985
Z_080 = 0.841621234


def analytic_rate(**kwargs: Any) -> Any:
    """Call the analytic route for a preference rate, absorbing its warning.

    The warning is asserted once, in its own test; everywhere else it is noise.
    """
    with pytest.warns(AnalyticProportionWarning):
        return power_analysis(statistic="preference_rate", method="analytic", **kwargs)


# --------------------------------------------------------------------------
# R1  with no clustering, this is the classical formula
# --------------------------------------------------------------------------


def test_unclustered_sample_size_matches_the_classical_formula() -> None:
    result = power_analysis(
        effect=0.5,
        sd=1.0,
        n_clusters=None,
        items_per_cluster=1,
        rho=0.0,
        power=0.80,
        alpha=0.05,
        method="analytic",
    )
    expected = (Z_0975 + Z_080) ** 2 * 1.0**2 / 0.5**2  # 31.3955...
    assert result.solved_for == "n_clusters"
    assert result.n_clusters.shape == (1,)
    assert result.n_clusters[0] == pytest.approx(expected, abs=1e-3)
    assert result.n_eff[0] == pytest.approx(expected, abs=1e-3)


def test_the_documented_gap_against_t_based_tools() -> None:
    """A z-based answer of 32, where G*Power and statsmodels say 34.

    The difference is the t correction those tools apply, and it is left
    unapplied here: the procedure this package computes power for is a bootstrap
    interval, not a t interval. The number is pinned so that someone comparing
    against another tool finds the discrepancy documented rather than surprising.
    """
    result = power_analysis(
        effect=0.5,
        sd=1.0,
        n_clusters=None,
        items_per_cluster=1,
        rho=0.0,
        power=0.80,
        alpha=0.05,
        method="analytic",
    )
    assert math.ceil(float(result.n_clusters[0])) == 32


# --------------------------------------------------------------------------
# R2  rho = 0 leaves the design effect at exactly one
# --------------------------------------------------------------------------


def test_design_effect_is_exactly_one_without_clustering() -> None:
    assert design_effect(items_per_cluster=3, rho=0.0)[()] == 1.0
    assert design_effect(items_per_cluster=1, rho=0.9)[()] == 1.0


def test_clustered_and_unclustered_paths_agree_exactly_at_rho_zero() -> None:
    """Not approximately: the same number, from the same code path.

    Forty clusters of three at rho = 0 carry as much information as 120
    independent items, so the two calls must not merely round to each other.
    """
    kwargs: dict[str, Any] = {
        "effect": 0.3,
        "sd": 1.0,
        "power": None,
        "rho": 0.0,
        "method": "analytic",
    }
    clustered = power_analysis(n_clusters=40, items_per_cluster=3, **kwargs)
    flat = power_analysis(n_clusters=120, items_per_cluster=1, **kwargs)
    assert clustered.solved_for == "power"
    assert clustered.power[0] == flat.power[0]


# --------------------------------------------------------------------------
# R3  the ceiling, and where the plan's own numbers come from
# --------------------------------------------------------------------------


def test_effective_n_reproduces_the_planned_band() -> None:
    assert effective_n(40, 3, 0.1)[()] == pytest.approx(100.0)
    assert effective_n(40, 3, 0.2)[()] == pytest.approx(120.0 / 1.4)


def test_items_per_cluster_cannot_pass_the_ceiling() -> None:
    ceiling = 40 / 0.2
    sizes = np.array([3.0, 15.0, 100.0, 1e6])
    reached = np.array([float(effective_n(40, m, 0.2)[()]) for m in sizes])
    assert np.all(np.diff(reached) > 0)
    assert np.all(reached < ceiling)
    assert reached[-1] == pytest.approx(ceiling, rel=1e-4)
    # The two figures the plan quotes for raising items from 3 to 15.
    assert reached[0] == pytest.approx(85.71, abs=0.01)
    assert reached[1] == pytest.approx(157.9, abs=0.05)


def test_an_unreachable_target_names_the_ceiling() -> None:
    with pytest.raises(ValueError, match="200"):
        power_analysis(
            effect=0.1,
            sd=1.0,
            n_clusters=40,
            items_per_cluster=None,
            rho=0.2,
            power=0.80,
            method="analytic",
        )


# --------------------------------------------------------------------------
# R4  the two directions invert each other
# --------------------------------------------------------------------------


def test_solving_forwards_and_backwards_round_trips() -> None:
    design: dict[str, Any] = {
        "sd": 1.0,
        "items_per_cluster": 3,
        "rho": 0.2,
        "power": 0.80,
        "method": "analytic",
    }
    mde = power_analysis(effect=None, n_clusters=40, **design)
    back = power_analysis(effect=float(mde.effect[0]), n_clusters=None, **design)
    assert mde.solved_for == "effect"
    assert back.n_clusters[0] == pytest.approx(40.0, rel=1e-6)


# --------------------------------------------------------------------------
# R5  the answer depends on the effect only through effect / sd
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 2.0, 17.0])
def test_power_is_invariant_and_mde_scales_with_sd(scale: float) -> None:
    base: dict[str, Any] = {
        "n_clusters": 40,
        "items_per_cluster": 3,
        "rho": 0.2,
        "method": "analytic",
    }
    unscaled = power_analysis(effect=0.4, sd=1.0, power=None, **base)
    scaled = power_analysis(effect=0.4 * scale, sd=scale, power=None, **base)
    assert scaled.power[0] == pytest.approx(float(unscaled.power[0]), rel=1e-12)

    mde_one = power_analysis(effect=None, sd=1.0, power=0.80, **base)
    mde_scaled = power_analysis(effect=None, sd=scale, power=0.80, **base)
    assert mde_scaled.effect[0] == pytest.approx(float(mde_one.effect[0]) * scale)


# --------------------------------------------------------------------------
# R6  the preference rate, against the exact binomial
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_preference_rate_power_tracks_the_exact_binomial() -> None:
    """At rho = 0 with no ties, the comparison is the exact binomial's power.

    Agreement is approximate on purpose. The procedure being simulated is a
    bootstrap interval, not the exact test, and the two differ by the bootstrap's
    own discreteness at finite n. The claim is that they land in the same place,
    within five points; a wider gap than that means the rate is being resampled
    wrongly, not that the two tests are merely different.
    """
    n, p, alpha = 200, 0.62, 0.05
    lower = binom.ppf(alpha / 2, n, 0.5)
    upper = binom.ppf(1 - alpha / 2, n, 0.5)
    exact = float(binom.cdf(lower - 1, n, p) + binom.sf(upper, n, p))

    simulated = power_analysis(
        statistic="preference_rate",
        effect=p,
        n_clusters=n,
        items_per_cluster=1,
        rho=0.0,
        power=None,
        alpha=alpha,
        method="simulation",
        n_sim=600,
        n_resamples=499,
        rng=20260909,
    )
    assert float(simulated.power[0]) == pytest.approx(exact, abs=0.05)


# --------------------------------------------------------------------------
# R7  consistency with paired_bootstrap
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_solved_mde_is_detected_at_the_requested_rate() -> None:
    """Generate data at the solved MDE and check paired_bootstrap finds it.

    **This is a correctness test, not a decision rule.** It asks one thing: if
    the returned MDE is right, then data generated at that effect must yield
    intervals excluding zero about `power` of the time. A failure means the
    implementation is wrong.

    It is not a test of which coverage route to take. That was settled by
    argument in docs/design/power_analysis.md -- simulate the procedure that will
    be run -- and the 0.03 tolerance below is a tolerance on Monte Carlo noise,
    not a threshold that decides anything. Both seeds are fixed, so a failure is
    reproducible and is never bad luck.
    """
    k, m, rho, target = 40, 3, 0.2, 0.80
    solved = power_analysis(
        effect=None,
        sd=1.0,
        n_clusters=k,
        items_per_cluster=m,
        rho=rho,
        power=target,
        alpha=0.05,
        method="simulation",
        n_sim=600,
        n_resamples=499,
        rng=20260909,
    )
    mde = float(solved.effect[0])

    # A second, independent run of the same procedure: nothing here reuses the
    # generator power_analysis drew from.
    rng = np.random.default_rng(717171)
    detected = 0
    trials = 600
    for _ in range(trials):
        diff, labels = clustered_differences(rng, k, m, mu=mde, rho=rho)
        interval = paired_bootstrap(diff, cluster=labels, n_resamples=499, rng=rng)
        detected += int(interval.ci_low > 0.0 or interval.ci_high < 0.0)

    empirical = detected / trials
    assert abs(empirical - target) <= 0.03


@pytest.mark.slow
def test_the_simulation_route_measures_its_own_error_rate() -> None:
    """The zero point of the power curve is the design's real error rate.

    It comes free with the curve, and it is the figure the study reports instead
    of the nominal alpha. At forty clusters it is expected to sit above the
    nominal 0.05 -- narrow intervals reject more often -- and the assertion is
    that it is measured at all and lands in the region bootstrap.py measured,
    not that it equals any particular value.
    """
    result = power_analysis(
        effect=None,
        sd=1.0,
        n_clusters=40,
        items_per_cluster=3,
        rho=0.2,
        power=0.80,
        alpha=0.05,
        method="simulation",
        n_sim=600,
        n_resamples=499,
        rng=20260909,
    )
    assert result.coverage_source == "measured"
    assert result.coverage_measured is not None
    assert result.curve is not None
    measured = float(result.coverage_measured[0])
    assert 0.88 <= measured <= 0.96
    assert measured < 0.95  # the shortfall bootstrap.py measured, not a repair


# --------------------------------------------------------------------------
# R8/R9  what the function refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"effect": 0.3, "power": 0.8}, "exactly one"),
        ({"effect": None, "power": None}, "exactly one"),
        ({"effect": 0.3, "power": None, "rho": 1.0}, "rho"),
        ({"effect": 0.3, "power": None, "alpha": 0.0}, "alpha"),
        ({"effect": None, "power": 1.0}, "power"),
        ({"effect": 0.0, "power": None}, "effect"),
        ({"effect": 0.3, "power": None, "sd": None}, "sd"),
        ({"effect": 0.3, "power": None, "tie_rate": 0.2}, "tie_rate"),
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "sd": 1.0,
        "n_clusters": 40,
        "items_per_cluster": 3,
        "rho": 0.2,
        "method": "analytic",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        power_analysis(**call)


def test_coverage_is_refused_on_the_simulation_route() -> None:
    """Two claims about one quantity, one of them carried in from elsewhere."""
    with pytest.raises(ValueError, match="coverage"):
        power_analysis(
            effect=0.3,
            sd=1.0,
            n_clusters=40,
            items_per_cluster=3,
            rho=0.2,
            power=None,
            coverage=0.927,
            method="simulation",
        )


def test_supplied_coverage_raises_the_analytic_power() -> None:
    """A real level below nominal is a larger alpha, and rejects more often.

    Named for the direction, like the warnings are: the power goes up and the
    MDE goes down. "Widens" would leave the reader to work out what widened.
    """
    base: dict[str, Any] = {
        "effect": 0.3,
        "sd": 1.0,
        "n_clusters": 40,
        "items_per_cluster": 3,
        "rho": 0.2,
        "power": None,
        "alpha": 0.05,
        "method": "analytic",
    }
    nominal = power_analysis(**base)
    supplied = power_analysis(coverage=0.927, **base)
    assert nominal.coverage_source == "nominal"
    assert supplied.coverage_source == "supplied"
    assert float(supplied.power[0]) > float(nominal.power[0])


def test_few_clusters_warns() -> None:
    with pytest.warns(FewClustersWarning):
        power_analysis(
            effect=0.3,
            sd=1.0,
            n_clusters=10,
            items_per_cluster=3,
            rho=0.2,
            power=None,
            method="analytic",
        )


def test_the_threshold_itself_does_not_warn() -> None:
    """MIN_CLUSTERS is the first count that is not warned about.

    Not the last count that is: a threshold tested from one side only
    drifts, because an off-by-one moves it and every one-sided test still
    passes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FewClustersWarning)
        power_analysis(
            effect=0.3,
            sd=1.0,
            n_clusters=MIN_CLUSTERS,
            items_per_cluster=3,
            rho=0.2,
            power=None,
            method="analytic",
        )


def test_the_analytic_route_warns_for_a_preference_rate() -> None:
    """And the warning has to say which way the error runs."""
    with pytest.warns(AnalyticProportionWarning, match="too small"):
        power_analysis(
            statistic="preference_rate",
            effect=0.62,
            n_clusters=40,
            items_per_cluster=3,
            rho=0.2,
            power=None,
            method="analytic",
        )


# --------------------------------------------------------------------------
# R10  monotonicity
# --------------------------------------------------------------------------


def test_power_rises_with_clusters_and_effect_and_falls_with_rho() -> None:
    base: dict[str, Any] = {
        "sd": 1.0,
        "items_per_cluster": 3,
        "power": None,
        "method": "analytic",
    }
    by_clusters = [
        float(power_analysis(effect=0.3, n_clusters=k, rho=0.2, **base).power[0])
        for k in (20, 40, 80)
    ]
    by_effect = [
        float(power_analysis(effect=d, n_clusters=40, rho=0.2, **base).power[0])
        for d in (0.1, 0.3, 0.5)
    ]
    by_rho = [
        float(power_analysis(effect=0.3, n_clusters=40, rho=r, **base).power[0])
        for r in (0.0, 0.1, 0.3)
    ]
    assert by_clusters == sorted(by_clusters)
    assert by_effect == sorted(by_effect)
    assert by_rho == sorted(by_rho, reverse=True)


def test_mde_falls_with_clusters_and_rises_with_ties() -> None:
    mdes = [
        float(
            power_analysis(
                effect=None,
                sd=1.0,
                n_clusters=k,
                items_per_cluster=3,
                rho=0.2,
                power=0.80,
                method="analytic",
            ).effect[0]
        )
        for k in (40, 80, 160)
    ]
    assert mdes == sorted(mdes, reverse=True)

    rates = [
        float(
            analytic_rate(
                effect=None,
                n_clusters=40,
                items_per_cluster=3,
                rho=0.2,
                power=0.80,
                tie_rate=t,
            ).effect[0]
        )
        for t in (0.0, 0.2, 0.4)
    ]
    assert rates == sorted(rates)  # further from the null of 0.5


# --------------------------------------------------------------------------
# R11  ties cost a share of the effective sample, not of the item count
# --------------------------------------------------------------------------


def test_ties_are_taken_off_the_effective_sample() -> None:
    """The distinction the two readings disagree about, in absolute terms.

    Applying (1 - tie_rate) to the nominal 120 items leaves 72. Applying it to
    the clustered effective sample of 85.71 leaves 51.43. The ratio between a
    tied and an untied design is the same either way, so only an absolute
    comparison separates them: the tied design has to match an untied one built
    at 51.43, and not the one built at 72.
    """
    tied = analytic_rate(
        effect=None,
        n_clusters=40,
        items_per_cluster=3,
        rho=0.2,
        power=0.80,
        tie_rate=0.4,
    )
    informative = 0.6 * 120.0 / 1.4  # 51.43
    matched = analytic_rate(
        effect=None,
        n_clusters=40,
        items_per_cluster=informative / 40.0,
        rho=0.0,
        power=0.80,
        tie_rate=0.0,
    )
    wrong = analytic_rate(
        effect=None,
        n_clusters=40,
        items_per_cluster=0.6 * 120.0 / 40.0,  # 72 items, the other reading
        rho=0.0,
        power=0.80,
        tie_rate=0.0,
    )
    assert float(tied.effect[0]) == pytest.approx(float(matched.effect[0]), rel=1e-9)
    assert float(tied.effect[0]) != pytest.approx(float(wrong.effect[0]), rel=1e-3)


# --------------------------------------------------------------------------
# Shape and provenance of the result
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.2, [0.2], [0.0, 0.1, 0.2, 0.3]])
def test_the_result_has_one_shape_whatever_rho_was(rho: Any) -> None:
    result = power_analysis(
        effect=None,
        sd=1.0,
        n_clusters=40,
        items_per_cluster=3,
        rho=rho,
        power=0.80,
        method="analytic",
    )
    expected = (np.asarray(rho, dtype=np.float64).size,)
    for field in (
        result.effect,
        result.power,
        result.n_clusters,
        result.items_per_cluster,
        result.n_items,
        result.n_eff,
        result.design_effect,
        result.n_eff_ceiling,
        result.rho,
    ):
        assert field.shape == expected


def test_a_sensitivity_range_is_ordered_and_finite_at_rho_zero() -> None:
    result = power_analysis(
        effect=None,
        sd=1.0,
        n_clusters=40,
        items_per_cluster=3,
        rho=[0.0, 0.1, 0.2, 0.3],
        power=0.80,
        method="analytic",
    )
    mde: NDArray[np.float64] = result.effect
    assert np.all(np.diff(mde) > 0)
    assert math.isinf(float(result.n_eff_ceiling[0]))
    assert result.curve is None  # a closed form needs no grid


def test_the_analytic_route_records_no_simulation_provenance() -> None:
    result = power_analysis(
        effect=None,
        sd=1.0,
        n_clusters=40,
        items_per_cluster=3,
        rho=0.2,
        power=0.80,
        method="analytic",
    )
    assert (result.n_sim, result.n_resamples, result.mc_se, result.seed) == (
        None,
        None,
        None,
        None,
    )
    assert result.statistic == "mean"
    assert result.alternative == "two-sided"
    assert result.variance_under == "null"


@pytest.mark.slow
def test_an_unseeded_simulation_records_the_seed_it_drew() -> None:
    result = power_analysis(
        effect=None,
        sd=1.0,
        n_clusters=40,
        items_per_cluster=3,
        rho=0.2,
        power=0.80,
        method="simulation",
        n_sim=100,
        n_resamples=99,
    )
    assert isinstance(result.seed, int)
    assert result.n_sim == 100
    assert result.mc_se is not None


def test_the_preference_rate_statistic_is_the_one_bootstrap_tests_use() -> None:
    """The two functions have to mean the same thing by a preference rate.

    Not a numerical claim: this pins the definition. The rate is the share of
    non-tied items favouring the first system, so a vector with four wins, one
    loss and three ties is 0.8, and the ties are absent from the denominator.
    """
    diff = np.array([1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    assert preference_rate(diff) == pytest.approx(0.8)
