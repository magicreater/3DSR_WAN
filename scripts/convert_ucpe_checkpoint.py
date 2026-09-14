"""Convert the pinned official UCPE relray_absmap adapter checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl3dsr.models.wan.geometry_conditioning import convert_official_ucpe_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official adapter-only Lightning checkpoint")
    parser.add_argument("destination", type=Path, help="new RL3dSR format-v2 checkpoint")
    parser.add_argument(
        "--assert-source-commit",
        required=True,
        help="asserted official UCPE source commit for this checkpoint",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            convert_official_ucpe_checkpoint(
                args.source,
                args.destination,
                asserted_source_commit=args.assert_source_commit,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
