"""On-policy distillation entrypoint.

  python scripts/train.py --config configs/distill.yaml [key=value ...]

Any DistillConfig field can be overridden from the CLI, e.g.:
  python scripts/train.py --config configs/distill.yaml lr=5e-6 steps=3000
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opd.trainer import DistillConfig, Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()

    raw = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        raw[key] = val

    cfg_kwargs = {}
    for key, val in raw.items():
        if key not in DistillConfig.__dataclass_fields__:
            raise SystemExit(f"unknown config key: {key}")
        default = getattr(DistillConfig, key)
        if isinstance(val, str) and not isinstance(default, str):
            if isinstance(default, bool):
                val = val.lower() in ("1", "true", "yes")
            elif isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
        cfg_kwargs[key] = val

    cfg = DistillConfig(**cfg_kwargs)
    print(cfg)
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
