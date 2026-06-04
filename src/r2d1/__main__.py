"""
r2d1 CLI

    python -m r2d1 watch <dir>  --job-id <id>  [--poll-every 30]
    python -m r2d1 pull  <uri>  --dest <dir>   [--force]
    python -m r2d1 secrets
"""
from __future__ import annotations

import argparse
import sys


def cmd_watch(args: argparse.Namespace) -> None:
    from r2d1 import Courier
    courier = Courier.from_env()
    print(f"[r2d1] watching {args.directory} for job {args.job_id}")
    courier.watch(args.directory, job_id=args.job_id,
                  poll_every=args.poll_every, blocking=True)


def cmd_pull(args: argparse.Namespace) -> None:
    from r2d1 import Fetcher
    info = Fetcher.from_env().pull(source=args.source, dest=args.dest,
                                   force=args.force)
    if not info.found:
        print("No checkpoint found — fresh start.")
        sys.exit(1)
    parts = [f"dir={info.local_dir}", f"cached={info.cached}",
             f"size={info.size_mb:.1f}MB"]
    if info.epoch is not None:
        parts.insert(0, f"epoch={info.epoch}")
    print("Ready: " + "  ".join(parts))


def cmd_secrets(args: argparse.Namespace) -> None:
    import os
    from r2d1.secrets import r2d1_config, _ALIASES

    cfg = r2d1_config()
    print("\n── r2d1 credentials ──────────────────────────────────────────")
    for cfg_key, aliases in _ALIASES.items():
        label = aliases[0]
        val   = cfg.get(cfg_key)
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
            print(f"  {label:<32}  {masked}")
        else:
            print(f"  {label:<32}  (not found)")
    print()


def main() -> None:
    p   = argparse.ArgumentParser(prog="r2d1",
              description="r2d1 — ML checkpoint courier (Cloudflare R2 + D1)")
    sub = p.add_subparsers(dest="command", metavar="command")

    # watch
    wp = sub.add_parser("watch", help="Watch a directory and ship checkpoints")
    wp.add_argument("directory")
    wp.add_argument("--job-id",      required=True, dest="job_id")
    wp.add_argument("--poll-every",  type=int, default=30, dest="poll_every")
    wp.set_defaults(func=cmd_watch)

    # pull
    pp = sub.add_parser("pull", help="Pull from R2 to local disk")
    pp.add_argument("source",
        help="r2://jobs/<id>/latest  |  r2://jobs/<id>/<name>  |  r2://<prefix>")
    pp.add_argument("--dest",  required=True)
    pp.add_argument("--force", action="store_true",
                    help="Re-download even if files exist on disk")
    pp.set_defaults(func=cmd_pull)

    # secrets
    sp = sub.add_parser("secrets", help="Show discovered credentials (masked)")
    sp.set_defaults(func=cmd_secrets)

    args = p.parse_args()
    if not args.command:
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
