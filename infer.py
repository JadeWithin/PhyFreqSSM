from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run full-scene inference with a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [sys.executable, "eval.py", "--config", args.config, "--checkpoint", args.checkpoint, "--split", "all"]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
