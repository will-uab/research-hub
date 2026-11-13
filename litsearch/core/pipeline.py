"""Run one stream: set up prices/DB/budget, execute the plugin, record the run."""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from ..streams import registry
from . import pagedata
from .budget import Budget, Prices, PricesMissing
from .config import Settings, example_prices
from .context import RunContext, RunOptions
from .db import DB
from .domains import DomainTagger

log = logging.getLogger(__name__)

__all__ = ["run", "RunOptions"]


def publish_page_data(settings: Settings, cfg: dict, db: DB, opts: RunOptions) -> None:
    """Regenerate the hub page's data after a stream finishes. Never allowed to mask the run's own outcome.
    Real runs write site/ (served over the tailnet); dry-runs write a preview to out/preview/."""
    out = settings.root / "out" / "preview" if opts.dry_run else settings.site_dir
    try:
        pagedata.regenerate(settings, cfg, db, out)
    except Exception:  # noqa: BLE001
        log.exception("page data generation failed")


def run(settings: Settings, cfg: dict, opts: RunOptions, stream: str = "literature") -> int:
    streams = registry()
    if stream not in streams:
        raise ValueError(f"unknown stream {stream!r}; available: {', '.join(sorted(streams))}")
    scfg = cfg["streams"].get(stream)
    if scfg is None:
        raise ValueError(f"no `streams.{stream}` block in interests.yaml")
    if not scfg.get("enabled", True) and not opts.dry_run:
        log.info("Stream %s is disabled in interests.yaml; nothing to do", stream)
        return 0

    now = datetime.now(settings.tz)
    today = now.date()
    log.info("Run %s [%s] (dry_run=%s, backfill=%s)", today, stream, opts.dry_run, opts.backfill)

    # ---- prices / key checks up front for real runs
    used_example_prices = False
    if opts.dry_run:
        price_env = {**example_prices(settings.root),
                     **{k: v for k, v in settings.env.items() if k.startswith("PRICE_") and v}}
        prices = Prices.from_env(price_env, stream)
        used_example_prices = "PRICE_INPUT_PER_MTOK" not in settings.env
    else:
        try:
            prices = Prices.from_env(settings.env, stream)
        except PricesMissing as e:
            raise RuntimeError(str(e)) from e
        if not settings.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing in .env")
        base_model = settings.get("LLM_MODEL") or "claude-haiku-5-5"
        if settings.model_for(scfg) != base_model and not settings.get(f"PRICE_{stream.upper()}_INPUT_PER_MTOK"):
            # Pricing a different model with the default model's prices would silently under-count spend.
            raise RuntimeError(f"{stream} uses model {settings.model_for(scfg)!r}, which differs from LLM_MODEL; "
                               f"set PRICE_{stream.upper()}_INPUT_PER_MTOK / _OUTPUT_PER_MTOK (and the cache prices) in .env")
        if stream == "literature" and "email" in settings.delivery and not settings.smtp_ready:
            raise RuntimeError("DELIVERY includes email but SMTP settings are incomplete in .env")

    # ---- state DB (dry-run works on a throwaway copy so nothing is recorded)
    tmpdir = None
    if opts.dry_run:
        if opts.dry_db:                                     # persistent preview DB (demos / tuning)
            db_path = Path(opts.dry_db)
            if not db_path.exists() and settings.db_path.exists():
                DB.copy_file(settings.db_path, db_path)
        else:
            tmpdir = tempfile.mkdtemp(prefix="lit-dry-")
            db_path = Path(tmpdir) / "seen.db"
            if settings.db_path.exists():
                DB.copy_file(settings.db_path, db_path)
    else:
        db_path = settings.db_path
    db = DB(db_path)

    try:
        weekly = settings.weekly_extra if stream == "journal_club" else None
        separate = tuple(k for k, v in cfg["streams"].items() if v.get("separate_budget"))
        budget = Budget(db, today.isoformat(), settings.daily_budget, prices, stream,
                        stream_cap=scfg.get("daily_cap_usd"), weekly_limit=weekly,
                        separate=bool(scfg.get("separate_budget")), exclude_streams=separate)
        ctx = RunContext(settings=settings, cfg=cfg, stream=stream, stream_cfg=scfg, db=db, opts=opts,
                         today=today, now=now, prices=prices, budget=budget,
                         tagger=DomainTagger(cfg["domains"]), used_example_prices=used_example_prices)
        run_id = db.start_run(stream)
        try:
            rc = streams[stream](ctx).execute()
        except Exception as e:  # noqa: BLE001 - recorded for the status banner, then re-raised
            db.finish_run(run_id, False, error=f"{type(e).__name__}: {e}", counts=ctx.counts, notes=ctx.notes)
            publish_page_data(settings, cfg, db, opts)
            raise
        db.finish_run(run_id, rc == 0, counts=ctx.counts, notes=ctx.notes)
        publish_page_data(settings, cfg, db, opts)
        return rc
    finally:
        db.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
