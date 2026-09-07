"""Statistical validity tooling for LLM and agent evaluations.

Every function here arrives with its assumptions written down and a test built
on a case whose answer is known independently of this code.
"""

from evalstat.bootstrap import (
    MIN_CLUSTERS,
    MIN_VALID_FRACTION,
    DegenerateResampleWarning,
    FewClustersWarning,
    PairedBootstrapResult,
    paired_bootstrap,
)

__version__ = "0.0.1"

__all__ = [
    "MIN_CLUSTERS",
    "MIN_VALID_FRACTION",
    "DegenerateResampleWarning",
    "FewClustersWarning",
    "PairedBootstrapResult",
    "paired_bootstrap",
]
