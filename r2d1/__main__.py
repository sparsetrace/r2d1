"""
r2d1 CLI

Usage:
    python -m r2d1 watch <directory> --job-id <id> [--poll-every 30]
    python -m r2d1 pull  --job-id <id> [--dest ./checkpoints]
    python -m r2d1 secrets
"""
from __future__ import annotations

import argparse
import sys


def cmd_watch(args: argparse.Namespace) -> None:
    from r2d1 import Courier
    courier = Courier.from_env()
    courier.watch(
        watch_dir   = args.directory,
        job_id      = args.job_id,
        poll_every  = args.poll_every,
        blocking    = True,          # CLI always blocks
    )


def cmd_pull(args: argparse.Namespace) -> None:
    from r2d1 import Restarter
    info = Restarter.from_env().pull(
        job_id = args.job_id,
        dest   = args.dest,
    )
    if info.found:
        print(f"Ready: epoch={info.epoch}  dir={info.local_dir}")
        sys.exit(0)
    else:
        print("No checkpoint found.")
        sys.exit(1)


def cmd_secrets(args: argparse.Namespace) -> None:
    from r2d1.secrets import r2d1_config, COMMON_OPTIONAL_SECRETS, export_secrets
    cfg = r2d1_config(discover_common=True)

    print("\n── R2D1 credentials ──────────────────────────────────────────")
    keys = [
        ("account_id",     "R2D1_ACCOUNT_ID"),
        ("r2_bucket",      "R2D1_R2_BUCKET"),
        ("r2_access_key",  "R2D1_R2_ACCESS_KEY"),
        ("r2_secret_key",  "R2D1_R2_SECRET_KEY"),
        ("r2_endpoint_url","R2D1_R2_ENDPOINT_URL"),
        ("api_token",      "R2D1_API_TOKEN"),
        ("d1_database_id", "R2D1_D1_DATABASE_ID"),
    ]
    for cfg_key, env_name in keys:
        val = cfg.get(cfg_key)
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
            print(f"  {env_name:<28} {masked}")
        else:
            print(f"  {env_name:<28} (not found)")

    print("\n── Common ML secrets ─────────────────────────────────────────")
    import os
    for name in COMMON_OPTIONAL_SECRETS:
        val = os.environ.get(name)
        status = "found" if val else "not found"
        print(f"  {name:<28} {status}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="r2d1",
        description="Lightweight ML checkpoint courier (Cloudflare R2 + D1)",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # watch
    watch_p = sub.add_parser("watch", help="Watch a directory and ship checkpoints to R2 + D1")
    watch_p.add_argument("directory",   help="Directory to watch for chk_*.json sidecars")
    watch_p.add_argument("--job-id",    required=True, dest="job_id")
    watch_p.add_argument("--poll-every", type=int, default=30, dest="poll_every",
                         help="Polling interval in seconds (default: 30)")
    watch_p.set_defaults(func=cmd_watch)

    # pull
    pull_p = sub.add_parser("pull", help="Pull latest checkpoint from R2 to local disk")
    pull_p.add_argument("--job-id",  required=True, dest="job_id")
    pull_p.add_argument("--dest",    default="./checkpoints",
                        help="Local destination directory (default: ./checkpoints)")
    pull_p.set_defaults(func=cmd_pull)

    # secrets
    sec_p = sub.add_parser("secrets", help="Show discovered credentials (masked)")
    sec_p.set_defaults(func=cmd_secrets)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
