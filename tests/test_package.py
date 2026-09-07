import evalstat


def test_version_is_exposed() -> None:
    assert isinstance(evalstat.__version__, str)
    assert evalstat.__version__.count(".") == 2


def test_public_api_is_covered_by_tests() -> None:
    # Guard rail: every entry added to __all__ must arrive with its own tests.
    assert set(evalstat.__all__) == {
        "MIN_CLUSTERS",
        "MIN_VALID_FRACTION",
        "DegenerateResampleWarning",
        "FewClustersWarning",
        "PairedBootstrapResult",
        "paired_bootstrap",
    }
