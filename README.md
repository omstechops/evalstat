# evalstat

Statistical validity tooling for LLM and agent evaluations.

`evalstat` is not an eval runner. Run your evals with whatever you already use
(Promptfoo, Braintrust, a notebook), then hand the per-example scores to
`evalstat` to answer the questions those tools leave open:

- How many examples does this comparison actually need?
- Is the gap between model A and model B distinguishable from noise?
- Across 12 prompt variants and 8 metrics, is the winner a winner or an artifact?
- Does the LLM judge agree with human labels well enough to be trusted?
- Does the average hide a subgroup where the system collapses?

## Status

Pre-alpha. The package skeleton is in place; no public functions are exported
yet. Each function ships together with reference-case tests.

## Design note: clustered evaluation sets

Existing eval tooling assumes examples are independent. Real eval sets usually
violate that: questions drawn from the same document, sessions from the same
user, utterances from the same recording or speaker. Treating clustered items as
independent inflates the effective sample size and reports confidence intervals
that are too narrow.

`paired_bootstrap()` therefore takes a `cluster=` argument and resamples
clusters rather than rows. This is the first place `evalstat` intends to differ
from the tools it sits next to, rather than restating them.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q
ruff check . && ruff format .
mypy
```

## License

MIT
