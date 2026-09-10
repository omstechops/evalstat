"""Tests for :func:`evalstat.agreement.judge_agreement`: the coefficient.

Written before the implementation, so this file is the specification the body has
to satisfy rather than a description of what it happened to do.

Every reference figure has **two witnesses**. The first is a cross-tabulation
small enough to work through by hand, whose ``p_o`` and ``p_e`` are written into
the test as literals with the arithmetic in its docstring -- the pattern of
``Z_0975`` in ``test_power.py``, and for the same reason: a reference value must
not come from the same code as the answer. The second is
``sklearn.metrics.cohen_kappa_score`` on the same data, which is a dev
dependency and a test oracle only. No coefficient here is copied from a paper.

This file is stage one of two. It covers the coefficient's arithmetic, which is
deterministic and fast; the clustered interval, which is stochastic and leans on
the shared resampler, is in ``test_agreement_interval.py``. The split is
deliberate: written together, a red interval test would not separate "the
coefficient is wrong" from "the interval is wrong".

``judge_agreement`` is imported from ``evalstat.agreement`` and not from
``evalstat`` because it is not exported until it works.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score

from evalstat.agreement import JudgeAgreementResult, Weights, judge_agreement

# These cases are about the point estimate, so the interval is bought at the
# lowest price that still exercises the resampling path.
POINT_RESAMPLES = 199
SEED = 20260910


def from_table(
    table: Sequence[Sequence[int]], categories: Sequence[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Expand a cross-tabulation into the two label vectors that produce it.

    Rows are the judge and columns the human, the orientation
    :class:`JudgeAgreementResult` documents, so the table written out in a test
    docstring and the table returned on the result are the same array.
    """
    judge: list[Any] = []
    human: list[Any] = []
    for i, row in enumerate(table):
        for j, count in enumerate(row):
            judge += [categories[i]] * count
            human += [categories[j]] * count
    return np.array(judge), np.array(human)


def sklearn_kappa(
    judge: NDArray[Any],
    human: NDArray[Any],
    categories: Sequence[Any],
    weights: Weights,
) -> float:
    """Compute the same coefficient with the second witness.

    **The weight matrix is not passed in, and must not be.** ``weights`` goes
    across as the string ``"linear"`` so that ``sklearn`` builds its own matrix
    from its own reading of the ordering. Handing it a matrix built here would
    leave that matrix untested and turn the oracle into an echo of the code it
    is supposed to check. ``labels`` is passed, because it is the caller's
    declared category set -- the same input the function under test receives,
    not a quantity derived from it -- and it fixes the order and size of the
    table on both sides.

    The two implementations disagree about normalisation: ``sklearn`` charges
    ``abs(i - j)`` where this package charges ``abs(i - j) / (k - 1)``. Kappa is
    a ratio of two sums against the same matrix, so the common factor cancels
    and the two agree exactly. That they agree *despite* weighting differently
    is part of what makes this a second witness rather than a copy.
    """
    return float(
        cohen_kappa_score(
            judge,
            human,
            labels=list(categories),
            weights=None if weights == "unweighted" else "linear",
        )
    )


def agree(
    judge: NDArray[Any],
    human: NDArray[Any],
    categories: Sequence[Any],
    weights: Weights,
    **kwargs: Any,
) -> JudgeAgreementResult:
    """Call under a fixed seed and a cheap resample count."""
    return judge_agreement(
        judge,
        human,
        categories=categories,
        weights=weights,
        n_resamples=POINT_RESAMPLES,
        rng=SEED,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# J1  unweighted kappa on a 2x2 table worked by hand
# ---------------------------------------------------------------------------


def test_j1_unweighted_two_by_two() -> None:
    """Pin the coefficient and both indices on a hand-worked 2x2 table.

    n = 50, rows the judge, columns the human::

                 human no   human yes   row
        judge no     20          5       25
        judge yes    10         15       25
        col          30         20

        p_o   = (20 + 15) / 50                             = 0.70
        p_e   = (25/50)(30/50) + (25/50)(20/50)
              = 0.50 * 0.60 + 0.50 * 0.40                  = 0.50
        kappa = (0.70 - 0.50) / (1 - 0.50)                 = 0.40
        PI    = |20 - 15| / 50                             = 0.10
        BI    = |5 - 10| / 50                              = 0.10
    """
    categories = ("no", "yes")
    judge, human = from_table(((20, 5), (10, 15)), categories)

    result = agree(judge, human, categories, "unweighted")

    assert result.p_observed == pytest.approx(0.70)
    assert result.p_expected == pytest.approx(0.50)
    assert result.kappa == pytest.approx(0.40)
    assert result.prevalence_index == pytest.approx(0.10)
    assert result.bias_index == pytest.approx(0.10)
    assert result.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "unweighted")
    )
    assert np.array_equal(result.table, np.array([[20, 5], [10, 15]]))
    assert result.n_items == 50


# ---------------------------------------------------------------------------
# J2  linear-weighted kappa, at three categories and at five
# ---------------------------------------------------------------------------


def test_j2_linear_weighted_three_categories() -> None:
    """Pin the weighted coefficient on a hand-worked preference table.

    Categories A < equal < B, so linear weights are ``abs(i - j) / 2``: a
    neighbouring disagreement costs 0.5 and the opposite verdict costs 1.
    n = 50, rows the judge::

                  A   equal   B    row
        A        10     4     1     15
        equal     3    12     3     18
        B         1     5    11     17
        col      14    21    15

        weighted disagreement observed
            0.5 * (4 + 3 + 3 + 5) + 1 * (1 + 1) = 9.5      -> 9.5 / 50 = 0.19
        p_o = 1 - 0.19                                     = 0.81

        expected counts E_ij = row_i * col_j / 50
            E_01 = 6.30  E_10 = 5.04  E_12 = 5.40  E_21 = 7.14
            E_02 = 4.50  E_20 = 4.76
        weighted disagreement expected
            0.5 * 23.88 + 1 * 9.26 = 21.20                 -> 21.2 / 50 = 0.424
        p_e = 1 - 0.424                                    = 0.576

        kappa = (0.81 - 0.576) / (1 - 0.576) = 0.234 / 0.424
    """
    categories = ("A", "equal", "B")
    table = ((10, 4, 1), (3, 12, 3), (1, 5, 11))
    judge, human = from_table(table, categories)

    result = agree(judge, human, categories, "linear")

    assert result.p_observed == pytest.approx(0.81)
    assert result.p_expected == pytest.approx(0.576)
    assert result.kappa == pytest.approx(0.234 / 0.424)
    assert result.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "linear")
    )
    assert np.array_equal(result.table, np.array(table))


def test_j2_linear_weighted_five_categories() -> None:
    """Pin the weighted coefficient on a symmetric five-point table.

    Categories 1 to 5, linear weights ``abs(i - j) / 4``. n = 100, diagonal 14
    everywhere, and six off-diagonal cells of 5: (1,2), (2,1), (4,5), (5,4) one
    step apart, and (3,5), (5,3) two steps apart. Row and column totals are
    (19, 19, 19, 19, 24), the same on both margins because the table is
    symmetric::

        weighted disagreement observed
            0.25 * (5 + 5 + 5 + 5) + 0.5 * (5 + 5) = 10    -> 10 / 100 = 0.10
        p_o = 0.90

        weighted disagreement expected
            (1/4) * sum_{i != j} |i - j| r_i r_j / 100^2
            sum_{i != j} |i - j| r_i r_j = 2 * 8170 = 16340
            (16340 / 4) / 10000                            = 0.4085
        p_e = 1 - 0.4085                                   = 0.5915

        kappa = (0.90 - 0.5915) / (1 - 0.5915) = 0.3085 / 0.4085

    The symmetry is doing a second job: equal margins make the bias index
    exactly zero, and an equal diagonal makes the prevalence index exactly zero,
    so this is also the reference point at which neither diagnostic fires.
    """
    categories = (1, 2, 3, 4, 5)
    table = np.zeros((5, 5), dtype=int)
    np.fill_diagonal(table, 14)
    for i, j in [(0, 1), (1, 0), (2, 4), (4, 2), (3, 4), (4, 3)]:
        table[i, j] = 5
    judge, human = from_table(table.tolist(), categories)

    result = agree(judge, human, categories, "linear")

    assert result.p_observed == pytest.approx(0.90)
    assert result.p_expected == pytest.approx(0.5915)
    assert result.kappa == pytest.approx(0.3085 / 0.4085)
    assert result.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "linear")
    )
    assert result.prevalence_index == pytest.approx(0.0)
    assert result.bias_index == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# J3  the two fixed points, as equalities rather than approximations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["unweighted", "linear"])
def test_j3_exact_agreement_is_exactly_one(weights: Weights) -> None:
    """Agreement on every item is 1.0 exactly, whatever the weighting.

    No disagreement means the observed weighted disagreement is zero, so kappa
    is ``1 - 0 / p_e`` for any positive ``p_e``. The weight scheme cannot reach
    it, which is the fixed point every weighted coefficient has to share.
    """
    categories = (1, 2, 3, 4, 5)
    labels = np.tile(np.array(categories), 6)

    result = agree(labels, labels.copy(), categories, weights)

    assert result.kappa == 1.0
    assert result.p_observed == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0


def test_j3_chance_model_reproduced_is_exactly_zero() -> None:
    """A table equal to the outer product of its own margins gives 0.0 exactly.

    Judge margins (20, 30, 50) and human margins (40, 40, 20) over n = 100 make
    every cell ``r_i * c_j / 100`` an integer::

              40   40   20
        20     8    8    4
        30    12   12    6
        50    20   20   10

    Observed equals expected cell by cell, so the weighted sums agree and kappa
    is zero for any weight matrix -- the chance model reproduced exactly, which
    is what the coefficient's zero means.
    """
    categories = ("low", "mid", "high")
    table = ((8, 8, 4), (12, 12, 6), (20, 20, 10))
    judge, human = from_table(table, categories)

    for weights in ("unweighted", "linear"):
        result = agree(judge, human, categories, weights)
        assert result.kappa == pytest.approx(0.0, abs=1e-12)
        assert result.p_observed == pytest.approx(result.p_expected)


# ---------------------------------------------------------------------------
# J4  invariances that hold by definition, so no oracle is needed
# ---------------------------------------------------------------------------


def test_j4_symmetric_in_the_two_raters() -> None:
    """Swapping the raters leaves everything but the table's orientation alone.

    ``p_o`` is symmetric, and ``p_e = sum w_ij r_i c_j`` is symmetric because
    the weight matrix is. The diagonal is unchanged by transposing, so the
    prevalence index is too, and the bias index sums the same absolute margin
    differences from the other side.

    The coefficient not caring which rater is which is exactly why the roles
    have to be carried by the caller: assumption 6 is about the study's
    arrangement, not about anything the arithmetic knows.
    """
    categories = ("A", "equal", "B")
    judge, human = from_table(((10, 4, 1), (3, 12, 3), (1, 5, 11)), categories)

    forward = agree(judge, human, categories, "linear")
    backward = agree(human, judge, categories, "linear")

    assert backward.kappa == forward.kappa
    assert backward.p_observed == forward.p_observed
    assert backward.p_expected == forward.p_expected
    assert backward.prevalence_index == forward.prevalence_index
    assert backward.bias_index == forward.bias_index
    assert np.array_equal(backward.table, forward.table.T)


def test_j4_labels_are_only_positions() -> None:
    """Relabelling categories without reordering them changes nothing.

    Distances are counted in positions in ``categories``, so the labels
    themselves carry no arithmetic. Strings and integers describing the same
    three-point scale have to give the same coefficient, the same indices and
    the same table.
    """
    table = ((10, 4, 1), (3, 12, 3), (1, 5, 11))
    worded = from_table(table, ("A", "equal", "B"))
    numbered = from_table(table, (-1, 0, 1))

    as_words = agree(*worded, ("A", "equal", "B"), "linear")
    as_numbers = agree(*numbered, (-1, 0, 1), "linear")

    assert as_numbers.kappa == as_words.kappa
    assert as_numbers.prevalence_index == as_words.prevalence_index
    assert as_numbers.bias_index == as_words.bias_index
    assert np.array_equal(as_numbers.table, as_words.table)


# ---------------------------------------------------------------------------
# J5  prevalence: a high match rate with a low coefficient
# ---------------------------------------------------------------------------


def test_j5_kappa_paradox_is_reported_as_prevalence() -> None:
    """Ninety per cent agreement, a coefficient under 0.45, and no bias at all.

    n = 100, rows the judge::

                 human no   human yes   row
        judge no     85          5       90
        judge yes     5          5       10
        col          90         10

        p_o   = (85 + 5) / 100                             = 0.90
        p_e   = 0.9 * 0.9 + 0.1 * 0.1                      = 0.82
        kappa = (0.90 - 0.82) / (1 - 0.82) = 0.08 / 0.18
        PI    = |85 - 5| / 100                             = 0.80
        BI    = |5 - 5| / 100                              = 0.00

    The point of the case is the pair of indices, not the coefficient: the two
    raters used one category for nearly everything, and the diagnostic says so
    with a large prevalence index and a bias index of exactly zero. Reporting
    0.44 on its own would read as a judge disagreeing with the human, which is
    not what this table shows.
    """
    categories = ("no", "yes")
    judge, human = from_table(((85, 5), (5, 5)), categories)

    result = agree(judge, human, categories, "unweighted")

    assert result.p_observed == pytest.approx(0.90)
    assert result.p_expected == pytest.approx(0.82)
    assert result.kappa == pytest.approx(0.08 / 0.18)
    assert result.prevalence_index == pytest.approx(0.80)
    assert result.bias_index == pytest.approx(0.0)
    assert result.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "unweighted")
    )


# ---------------------------------------------------------------------------
# J6  bias: the mirror image, where the judge is systematically harsher
# ---------------------------------------------------------------------------


def test_j6_systematic_harshness_is_reported_as_bias() -> None:
    """A judge that says "no" far less often than the human does.

    n = 100, rows the judge::

                 human no   human yes   row
        judge no     30          5       35
        judge yes    30         35       65
        col          60         40

        p_o   = (30 + 35) / 100                            = 0.65
        p_e   = 0.35 * 0.60 + 0.65 * 0.40                   = 0.47
        kappa = (0.65 - 0.47) / (1 - 0.47) = 0.18 / 0.53
        PI    = |30 - 35| / 100                            = 0.05
        BI    = |5 - 30| / 100                             = 0.25

    Read against J5 this is the case for reporting both indices. The two tables
    have similar coefficients and opposite diagnoses: J5 is prevalence with no
    bias, this is bias with almost no prevalence. One is a rubric that does not
    discriminate, the other is a judge whose threshold sits in the wrong place
    and which recalibrating could fix, and the coefficient alone cannot tell
    them apart.
    """
    categories = ("no", "yes")
    judge, human = from_table(((30, 5), (30, 35)), categories)

    result = agree(judge, human, categories, "unweighted")

    assert result.p_observed == pytest.approx(0.65)
    assert result.p_expected == pytest.approx(0.47)
    assert result.kappa == pytest.approx(0.18 / 0.53)
    assert result.prevalence_index == pytest.approx(0.05)
    assert result.bias_index == pytest.approx(0.25)
    assert result.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "unweighted")
    )


# ---------------------------------------------------------------------------
# J7  categories come from the design, and inferring them fails intermittently
# ---------------------------------------------------------------------------

# A five-point scale the raters used only at 1, 2 and 5. n = 70.
UNEVEN_TABLE = ((20, 5, 1), (4, 15, 2), (1, 3, 19))
UNEVEN_USED = (1, 2, 5)

# A five-point scale the raters used at 1 to 4. n = 84.
EVEN_TABLE = ((15, 4, 1, 0), (3, 14, 4, 1), (1, 4, 14, 3), (0, 1, 4, 15))
EVEN_USED = (1, 2, 3, 4)

FIVE_POINT = (1, 2, 3, 4, 5)


def test_j7_dropping_unused_labels_moves_an_unevenly_spaced_scale() -> None:
    """Declaring the design's scale is not the same as declaring what occurred.

    The raters used 1, 2 and 5 of a five-point scale. Declared in full, the
    distances are 0.25, 0.75 and 1.00; with the unused labels dropped they
    become 0.50, 0.50 and 1.00, which is a different scale and a different
    coefficient. Nothing in the output announces the substitution, which is why
    ``categories`` is required rather than inferred.
    """
    judge, human = from_table(UNEVEN_TABLE, UNEVEN_USED)

    declared = agree(judge, human, FIVE_POINT, "linear")
    inferred = agree(judge, human, UNEVEN_USED, "linear")

    assert declared.kappa != pytest.approx(inferred.kappa)
    assert declared.kappa == pytest.approx(
        sklearn_kappa(judge, human, FIVE_POINT, "linear")
    )
    assert inferred.kappa == pytest.approx(
        sklearn_kappa(judge, human, UNEVEN_USED, "linear")
    )
    assert declared.table.shape == (5, 5)
    assert inferred.table.shape == (3, 3)


def test_j7_dropping_a_trailing_label_leaves_the_coefficient_alone() -> None:
    """The same mistake, invisible this time -- which is the worse half.

    Here the raters used 1 to 4, so dropping the unused 5 rescales every
    distance by the same factor, the factor cancels between the observed and
    expected sums, and the coefficient does not move at all. A project that
    infers its categories is therefore right some of the time for a reason that
    has nothing to do with its rating design.

    The table and the prevalence index move in both cases, which is the only
    signal there is, and it is a signal about the report rather than about the
    coefficient.
    """
    judge, human = from_table(EVEN_TABLE, EVEN_USED)

    declared = agree(judge, human, FIVE_POINT, "linear")
    inferred = agree(judge, human, EVEN_USED, "linear")

    assert declared.kappa == pytest.approx(inferred.kappa, abs=1e-12)
    assert declared.table.shape == (5, 5)
    assert inferred.table.shape == (4, 4)
    # The declared scale has an unused category, so its diagonal holds a zero
    # and the prevalence index is the largest diagonal cell rather than a range.
    assert declared.prevalence_index != pytest.approx(inferred.prevalence_index)


def test_j7_order_carries_the_scale() -> None:
    """The same labels in a different order are a different scale.

    ``(2, 1, 5)`` is a legal category list and a scale nobody rated on. It has
    to give a different coefficient, because the distances it declares are
    different ones.
    """
    judge, human = from_table(UNEVEN_TABLE, UNEVEN_USED)

    ordered = agree(judge, human, UNEVEN_USED, "linear")
    misordered = agree(judge, human, (2, 1, 5), "linear")

    assert ordered.kappa != pytest.approx(misordered.kappa)


# ---------------------------------------------------------------------------
# J8  the two weight schemes are different quantities
# ---------------------------------------------------------------------------


def test_j8_weight_schemes_are_not_comparable() -> None:
    """Two coefficients from one table, and neither means anything alone.

    Every disagreement in this table is one step wide, which is the case
    weighting exists for. n = 80, rows the judge, categories 1 < 2 < 3::

               1    2    3    row
        1     20    5    0     25
        2      5   20    5     30
        3      0    5   20     25
        col   25   30   25

        unweighted  p_o = 60 / 80                          = 0.75
                    p_e = (625 + 900 + 625) / 6400         = 0.3359375
                    kappa = 0.4140625 / 0.6640625

        linear      observed disagreement 0.5 * 20 / 80    = 0.125
                    expected disagreement 2750 / 6400      = 0.4296875
                    kappa = 0.3046875 / 0.4296875

    The linear figure is the higher of the two because every disagreement here
    is adjacent and linear weighting charges half price for it. A report that
    quotes 0.71 without saying "linear" has quoted a number that cannot be
    compared to the 0.62 someone else computed from the same table.
    """
    categories = (1, 2, 3)
    judge, human = from_table(((20, 5, 0), (5, 20, 5), (0, 5, 20)), categories)

    unweighted = agree(judge, human, categories, "unweighted")
    linear = agree(judge, human, categories, "linear")

    assert unweighted.kappa == pytest.approx(0.4140625 / 0.6640625)
    assert linear.kappa == pytest.approx(0.3046875 / 0.4296875)
    assert linear.kappa > unweighted.kappa
    assert unweighted.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "unweighted")
    )
    assert linear.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "linear")
    )
    assert unweighted.weights == "unweighted"
    assert linear.weights == "linear"
    assert tuple(linear.categories) == categories


# ---------------------------------------------------------------------------
# J9  a degenerate observed table is a fact about the data, so it raises
# ---------------------------------------------------------------------------


def test_j9_one_constant_rater_is_exactly_zero_and_not_an_error() -> None:
    """A judge that answered "equal" every time scores exactly 0, not an error.

    This is the half of the degenerate case that is easy to get wrong, so it is
    pinned first. With the judge constant at one category the observed
    disagreement and the expected disagreement are the same sum term by term --
    the joint distribution *is* the product of the marginals, because a point
    mass is independent of everything -- so the ratio is one and kappa is
    exactly zero.

    Zero is the right answer and a meaningful one: a rater who always says the
    same thing agrees exactly as often as chance predicts, which is what the
    coefficient is built to report. Raising here would throw away a correct
    measurement, and the diagnostics say what happened -- the table has one
    non-empty row.
    """
    categories = ("A", "equal", "B")
    judge = np.array(["equal"] * 30)
    human = np.array(["A", "equal", "B"] * 10)

    result = agree(judge, human, categories, "linear")

    assert result.kappa == 0.0
    assert result.kappa == pytest.approx(
        sklearn_kappa(judge, human, categories, "linear")
    )
    assert np.count_nonzero(result.table.sum(axis=1)) == 1


def test_j9_two_constant_raters_on_different_categories_is_also_zero() -> None:
    """Still not degenerate: two point masses, one apart, still give exactly 0.

    The boundary matters because it says what the degenerate case actually is.
    Both raters constant is not enough; a table with its whole mass in one
    off-diagonal cell has a non-zero expected disagreement equal to its observed
    one, so the coefficient is defined and zero.
    """
    categories = ("A", "equal", "B")
    judge = np.array(["A"] * 30)
    human = np.array(["B"] * 30)

    result = agree(judge, human, categories, "linear")

    assert result.kappa == 0.0
    assert result.p_observed == 0.0


def test_j9_both_raters_on_one_category_raises_naming_the_state() -> None:
    """The one degenerate table there is, and the message has to say which.

    ``sum(w * outer(r, c))`` is zero only where the weight is zero wherever the
    two marginals have mass together, and the weight is zero only on the
    diagonal, so the denominator vanishes exactly when both raters used one and
    the same category throughout. Then the chance model already expects perfect
    agreement, the raters deliver it, and there is nothing left for the
    coefficient to measure: 0/0.

    It is an error rather than a NaN because it is a fact about the data -- a
    rubric dimension nobody discriminated on -- and a NaN invites being read as
    a number. The message names the state in those words: both raters, that
    category, expected agreement 1, kappa undefined.
    """
    categories = ("A", "equal", "B")
    labels = np.array(["equal"] * 30)

    with pytest.raises(ValueError, match="both raters"):
        agree(labels, labels.copy(), categories, "linear")
    with pytest.raises(ValueError, match="equal"):
        agree(labels, labels.copy(), categories, "linear")
    with pytest.raises(ValueError, match="undefined"):
        agree(labels, labels.copy(), categories, "linear")


# ---------------------------------------------------------------------------
# J10  input rejection
# ---------------------------------------------------------------------------

PAIRED = (np.array([0, 1, 2] * 10), np.array([0, 1, 1] * 10))
THREE = (0, 1, 2)

BAD_CALLS: list[tuple[str, dict[str, Any]]] = [
    (
        "judge and human of different lengths",
        {"judge": [0, 1, 2], "human": [0, 1], "categories": THREE},
    ),
    (
        "cluster of the wrong length",
        {
            "judge": PAIRED[0],
            "human": PAIRED[1],
            "categories": THREE,
            "cluster": np.arange(5),
        },
    ),
    (
        "ceiling of the wrong length",
        {
            "judge": PAIRED[0],
            "human": PAIRED[1],
            "categories": THREE,
            "ceiling": [0, 1, 2],
        },
    ),
    (
        "a label absent from categories",
        {"judge": PAIRED[0], "human": PAIRED[1], "categories": (0, 1)},
    ),
    (
        "categories repeating a label",
        {"judge": PAIRED[0], "human": PAIRED[1], "categories": (0, 1, 1, 2)},
    ),
    (
        "a single category",
        {"judge": [0] * 30, "human": [0] * 30, "categories": (0,)},
    ),
    (
        "an unknown weight scheme",
        {
            "judge": PAIRED[0],
            "human": PAIRED[1],
            "categories": THREE,
            "weights": "quadratic",
        },
    ),
    (
        "no resamples",
        {"judge": PAIRED[0], "human": PAIRED[1], "categories": THREE, "n_resamples": 0},
    ),
    (
        "a confidence level of one",
        {
            "judge": PAIRED[0],
            "human": PAIRED[1],
            "categories": THREE,
            "confidence_level": 1.0,
        },
    ),
    (
        "an unknown interval method",
        {
            "judge": PAIRED[0],
            "human": PAIRED[1],
            "categories": THREE,
            "method": "studentised",
        },
    ),
    (
        "a single cluster",
        {
            "judge": PAIRED[0],
            "human": PAIRED[1],
            "categories": THREE,
            "cluster": np.zeros(30, dtype=int),
        },
    ),
]


@pytest.mark.parametrize(
    ("kwargs",), [(call,) for _, call in BAD_CALLS], ids=[name for name, _ in BAD_CALLS]
)
def test_j10_malformed_input_raises_value_error(kwargs: dict[str, Any]) -> None:
    """Every one of these is a call that cannot mean anything.

    ``weights`` defaults to ``linear`` here only so that each case tests the one
    thing it names; the function itself has no default, which is J10's other
    test.
    """
    call = {"weights": "linear", **kwargs}
    with pytest.raises(ValueError):
        judge_agreement(**call)


def test_j10_categories_and_weights_have_no_defaults() -> None:
    """Both are keyword-only and required, so omitting either is a TypeError.

    This is D2 and D4 enforced by the signature rather than by a check inside
    the body: the caller cannot fall into ``unweighted``, and cannot fall into
    a category set inferred from the data, because neither call is legal.
    """
    judge, human = PAIRED
    with pytest.raises(TypeError):
        judge_agreement(judge, human, weights="linear")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        judge_agreement(judge, human, categories=THREE)  # type: ignore[call-arg]
