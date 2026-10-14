"""Combine metrics files into the results tables (design doc §24).

    uv run python scripts/make_results_table.py outputs/metrics/*/data_exp/test.json

Refuses to combine files computed on different sample sets (manifest hashes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prs.metrics.tables import ManifestMismatchError, results_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics JSON files")
    parser.add_argument("--out", type=Path, help="also write the Markdown here")
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.metrics]
    try:
        markdown = results_markdown(reports)
    except ManifestMismatchError as error:
        raise SystemExit(f"error: {error}") from None
    print(markdown)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
