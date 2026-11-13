"""Command line: `lit [--dry-run] [--backfill N]` and `lit feedback +PMID ...`."""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys

from .core import config, pipeline
from .core.context import RunOptions
from .core.db import DB
from .streams import registry
from .streams.literature import report_failure

log = logging.getLogger("litsearch")


def setup_logging(root=config.ROOT, verbose: bool = False) -> None:
    (root / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(root / "logs" / "litsearch.log", maxBytes=1_000_000,
                                              backupCount=5, encoding="utf-8")
    sh = logging.StreamHandler(sys.stderr)
    for h in (fh, sh):
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[fh, sh], force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def cmd_feedback(codes: list[str]) -> int:
    settings = config.load_settings(require_env=False)
    if not codes:
        print("usage: lit feedback +PMID | -PMID [...]   (arXiv: +arxiv:2401.01234, others: +doi:10.x/y)")
        return 2
    db = DB(settings.db_path)
    rc = 0
    for raw in codes:
        if raw[:1] not in "+-" or len(raw) < 2:
            print(f"skip {raw!r}: start with + (like) or - (dislike)")
            rc = 1
            continue
        sign, code = (1 if raw[0] == "+" else -1), raw[1:]
        row = db.resolve_code(code)
        if row is None:
            print(f"not found: {code} (only papers already in the database can be rated)")
            rc = 1
            continue
        db.add_feedback(row["id"], sign)
        print(f"{'liked' if sign > 0 else 'disliked'}: {row['title']}")
    db.close()
    return rc


def cmd_migrate_config() -> int:
    settings = config.load_settings(require_env=False)
    diff = config.migrate_file(settings.interests_path)
    if not diff:
        print(f"{settings.interests_path} is already version 2; nothing to do.")
        return 0
    print(diff)
    print(f"\nMigrated. Original saved as {settings.interests_path.with_name('interests.v1.yaml.bak')}")
    return 0


def cmd_build_site() -> int:
    """Regenerate site/ data from the database without running any stream."""
    from .core import pagedata
    settings = config.load_settings(require_env=False)
    cfg = config.load_interests(settings.interests_path)
    db = DB(settings.db_path)
    stats = pagedata.regenerate(settings, cfg, db, settings.site_dir)
    db.close()
    print(f"Wrote {settings.site_dir}: {stats['items']} items ({stats['recent']} recent), {stats['shards']} abstract shard(s)")
    return 0


def cmd_gen_systemd() -> int:
    """Write per-stream .service/.timer files from the schedules in interests.yaml, and print the install commands."""
    import getpass
    from .core import systemd
    settings = config.load_settings(require_env=False)
    cfg = config.load_interests(settings.interests_path)
    files = systemd.render_units(cfg, settings.root, getpass.getuser())
    out = settings.root / "systemd"
    out.mkdir(exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    print(f"Wrote {len(files)} unit files to {out}:\n  " + "\n  ".join(sorted(files)))
    print("\nInstall (needs sudo; includes the web server unit, which now also serves the JSON API):\n")
    print(systemd.install_commands(files, settings.root))
    return 0


def cmd_watch(args: list[str]) -> int:
    """lit watch list | search NAME | add NAME [--profile-id N] [--openalex-id A...] | remove NAME | resolve"""
    from .core import watch
    settings = config.load_settings(require_env=False)
    sub = args[0] if args else "list"
    if sub == "list":
        for p in watch.load(settings):
            print(f"  {p['name']:<24} reporter={p.get('profile_id', '-'):<9} openalex={p.get('openalex_id', '-')}")
        return 0
    if sub == "resolve":
        for r in watch.resolve_missing(settings):
            print(f"  {r['status']:<10} {r['name']:<24} {r['detail']}")
        return 0
    if sub == "search" and len(args) > 1:
        res = watch.search_candidates(" ".join(args[1:]))
        print("RePORTER PI profiles:")
        for c in res["reporter"]:
            print(f"  profile {c['profile_id']:<9} {c['name']:<26} {c['awards']:>3} awards  {'; '.join(c['institutions'])}")
        print("OpenAlex authors:")
        for c in res["openalex"]:
            print(f"  {c['id']:<13} {c['name']:<26} {c['works']:>4} works  {'; '.join(c['institutions'])}")
        for e in res["errors"]:
            print("  !", e)
        return 0
    if sub == "add" and len(args) > 1:
        name, rest = args[1], args[2:]
        entry = {"name": name}
        for flag, key in (("--profile-id", "profile_id"), ("--openalex-id", "openalex_id")):
            if flag in rest:
                entry[key] = rest[rest.index(flag) + 1]
        _, added = watch.add(settings, entry)
        print(("added " if added else "already on the list (ids merged): ") + name)
        return 0
    if sub == "remove" and len(args) > 1:
        _, n = watch.remove(settings, name=args[1])
        print(f"removed {n}")
        return 0 if n else 1
    print(cmd_watch.__doc__)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "feedback":      # manual: codes like -123 or -arxiv:... look like flags
        setup_logging()
        return cmd_feedback(argv[1:])
    if argv and argv[0] == "watch":
        setup_logging()
        return cmd_watch(argv[1:])
    if argv and argv[0] == "gen-systemd":
        setup_logging()
        return cmd_gen_systemd()
    if argv and argv[0] == "build-site":
        setup_logging()
        return cmd_build_site()
    if argv and argv[0] == "migrate-config":
        setup_logging()
        return cmd_migrate_config()

    ap = argparse.ArgumentParser(prog="lit", description="Daily literature digest.")
    ap.add_argument("--stream", default="literature", metavar="NAME",
                    help="stream to run: " + ", ".join(sorted(registry())) + " (default: literature)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write out/digest-DATE.html, print cost estimate; no email, no API calls, nothing recorded")
    ap.add_argument("--backfill", type=int, metavar="N", help="look back N days instead of the default window")
    ap.add_argument("--max-candidates", type=int, help="override the LLM candidate cap")
    ap.add_argument("--force", action="store_true", help="tuning: run even if too few new ratings have accumulated")
    ap.add_argument("--redo", action="store_true", help="podcast: replace today's episode (another model call)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)
    settings = None
    try:
        settings = config.load_settings(require_env=not args.dry_run)
        cfg = config.load_interests(settings.interests_path)
        return pipeline.run(settings, cfg, RunOptions(args.dry_run, args.backfill, args.max_candidates, force=args.force, redo=args.redo), args.stream)
    except Exception as e:  # noqa: BLE001
        log.exception("Run failed")
        if settings is not None and not args.dry_run and args.stream == "literature":
            report_failure(settings, f"{type(e).__name__}: {e}")
        else:
            print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
