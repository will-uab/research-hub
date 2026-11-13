"""Settings (.env), and interests.yaml v2 loading + v1 -> v2 migration."""
from __future__ import annotations

import difflib
import logging
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DAILY_BUDGET = 0.15
DEFAULT_WEEKLY_EXTRA = 0.10


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (comments, blank lines, optional quotes)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def check_env_permissions(path: Path) -> str | None:
    """Return a warning string if .env is readable by group/others."""
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"{path} has permissions {oct(mode)}; run: chmod 600 {path}"
    return None


@dataclass
class Settings:
    env: dict[str, str]
    root: Path
    has_env_file: bool = False

    def get(self, key: str, default: str = "") -> str:
        return self.env.get(key, default)

    def float(self, key: str) -> float | None:
        v = self.env.get(key, "").strip()
        return float(v) if v else None

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.get("LIT_TZ", "America/Chicago"))

    @property
    def db_path(self) -> Path:
        return self.root / "data" / "seen.db"

    @property
    def interests_path(self) -> Path:
        return self.root / "config" / "interests.yaml"

    @property
    def site_dir(self) -> Path:
        return self.root / "site"

    @property
    def daily_budget(self) -> float:
        return self.float("DAILY_BUDGET_USD") or DEFAULT_DAILY_BUDGET

    @property
    def weekly_extra(self) -> float:
        v = self.float("WEEKLY_EXTRA_USD")
        return DEFAULT_WEEKLY_EXTRA if v is None else v

    def model_for(self, stream_cfg: dict) -> str:
        """Model id: the env var named by the stream's `model_env`, else LLM_MODEL, else the Haiku default."""
        return (self.get(stream_cfg.get("model_env", "LLM_MODEL")) or self.get("LLM_MODEL")
                or "claude-haiku-5-5")

    def effort_for(self, stream: str) -> str:
        k = f"LLM_EFFORT_{stream.upper()}"
        return self.get(k) if k in self.env else self.get("LLM_EFFORT", "")

    @property
    def delivery(self) -> set[str]:
        """Subset of {'web', 'email'} from DELIVERY=web|email|both (default web)."""
        v = self.get("DELIVERY", "web").strip().lower()
        if v not in ("web", "email", "both"):
            raise ValueError(f"DELIVERY must be web, email or both (got {v!r})")
        return {"web", "email"} if v == "both" else {v}

    @property
    def smtp_ready(self) -> bool:
        return all(self.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM", "MAIL_TO"))


def load_settings(root: Path = ROOT, require_env: bool = True) -> Settings:
    """Load .env (if present) overlaid by real environment variables."""
    env: dict[str, str] = {}
    env_path = root / ".env"
    has = env_path.exists()
    if has:
        warn = check_env_permissions(env_path)
        if warn:
            log.warning(warn)
        env.update(parse_env_file(env_path))
    elif require_env:
        raise FileNotFoundError(f"{env_path} not found; copy .env.example to .env and fill it in")
    for k, v in os.environ.items():
        if k.startswith(("ANTHROPIC_", "LLM_", "PRICE_", "DAILY_", "WEEKLY_", "NCBI_", "SMTP_", "MAIL_", "LIT_",
                         "DELIVERY")):
            env[k] = v
    return Settings(env=env, root=root, has_env_file=has)


def example_prices(root: Path = ROOT) -> dict[str, str]:
    """Dry-run fallback only: read prices from .env.example when .env is absent."""
    p = root / ".env.example"
    return parse_env_file(p) if p.exists() else {}


# ====================================================================== interests.yaml
V1_KEYS = ("profile", "groups", "major", "medrxiv", "arxiv", "exclude", "limits", "digest")

# Where each v1 query group lives in v2, and every domain it should tag.
GROUP_DOMAINS: dict[str, list[str]] = {
    "llm_clinical_extraction": ["llm_ai"],
    "llm_evaluation": ["llm_ai"],
    "agentic_research_workflows": ["llm_ai"],
    "llm_general_medicine": ["llm_ai"],
    "ai_prediction_id_addiction": ["llm_ai", "hiv_addiction", "infectious_diseases"],
    "hiv_care_continuum": ["hiv_addiction"],
    "oud_outcomes_infections": ["hiv_addiction", "infectious_diseases"],
}
DOMAIN_ORDER = ["llm_ai", "infectious_diseases", "hiv_addiction"]

# Domain-level terms: used for domain tagging and by the grants / trials / awards streams.
# (The AND-structured literature_groups below remain the precise PubMed queries.)
DOMAIN_SEEDS: dict[str, dict] = {
    "llm_ai": {
        "label": "LLM / AI", "weight": 10,
        "keywords": ["large language model*", "LLM", "LLMs", "GPT", "ChatGPT", "generative AI", "foundation model*",
                     "natural language processing", "clinical NLP", "machine learning", "deep learning",
                     "artificial intelligence", "AI agent*", "agentic", "chatbot*", "predictive model*"],
        "mesh": ["Large Language Models", "Natural Language Processing", "Artificial Intelligence",
                 "Machine Learning", "Deep Learning"],
    },
    "infectious_diseases": {
        "label": "Infectious Diseases", "weight": 4,
        "keywords": ["infectious disease*", "infection*", "sepsis", "bacteremia", "endocarditis", "antimicrobial*",
                     "antibiotic*", "antimicrobial resistance", "tuberculosis", "hepatitis", "Clostridioides difficile",
                     "COVID-19", "SARS-CoV-2", "influenza", "vaccin*", "syphilis", "gonorrhea", "chlamydia",
                     "sexually transmitted", "fungal infection*", "osteomyelitis", "antimicrobial stewardship"],
        "mesh": ["Communicable Diseases", "Bacterial Infections", "Sepsis", "Bacteremia", "Endocarditis",
                 "Anti-Bacterial Agents", "Drug Resistance, Microbial", "Tuberculosis", "Hepatitis C", "COVID-19",
                 "Vaccines", "Sexually Transmitted Diseases", "Mycoses"],
    },
    "hiv_addiction": {
        "label": "HIV + Addiction", "weight": 5,
        "keywords": ["HIV", "AIDS", "PrEP", "pre-exposure prophylaxis", "antiretroviral*", "opioid use disorder",
                     "OUD", "MOUD", "buprenorphine", "methadone", "naltrexone", "overdose", "substance use",
                     "addiction", "people who inject drugs", "injection drug use", "fentanyl", "xylazine",
                     "alcohol use disorder", "methamphetamine", "stimulant use"],
        "mesh": ["HIV Infections", "Pre-Exposure Prophylaxis", "Anti-HIV Agents", "Opioid-Related Disorders",
                 "Opiate Substitution Treatment", "Substance-Related Disorders", "Drug Overdose",
                 "Substance Abuse, Intravenous", "Alcohol Drinking"],
    },
}

STREAM_DEFAULTS: dict[str, dict] = {
    "trials": {
        "enabled": True, "schedule": "04:10", "model_env": "LLM_MODEL", "window_days": 2,
        "max_candidates": 40, "batch_size": 15, "daily_cap_usd": 0.04,
        "ai_terms": ["large language model*", "LLM", "artificial intelligence", "machine learning", "deep learning",
                     "chatbot*", "natural language processing", "clinical decision support", "predictive model*"],
        "track_statuses": ["NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING",
                           "SUSPENDED", "TERMINATED", "COMPLETED", "WITHDRAWN"],
    },
    "grants": {
        "enabled": True, "schedule": "04:20", "model_env": "LLM_MODEL", "window_days": 3,
        "max_candidates": 40, "batch_size": 15, "daily_cap_usd": 0.03, "min_relevance": 4,
        "fallback_max_items": 10, "refresh_limit": 120, "weekly_summary": "friday", "deadline_flag_days": 60,
        # Exact Grants.gov sub-agency codes (a parent like "HHS" is not filterable). HHS-AHRQ currently has no
        # open opportunities; it is listed so it is picked up when it does.
        "agencies": ["HHS-NIH11", "HHS-AHRQ", "HHS-CDC-NCHHSTP", "HHS-CDC-NCEZID", "HHS-CDC-NCIPC",
                     "HHS-CDC-NCCDPHP", "HHS-HRSA", "HHS-SAMHS-SAMHSA"],
        "institutes": ["NIDA", "NIAID", "NIAAA", "NIMH", "NLM", "NIMHD"],
        "mechanisms": ["R01", "R21", "R34", "K*", "U*"],
        "strict_preferences": False,
        # An issuing institute implies a domain even when no keyword matches (the LLM still judges fit).
        "institute_domains": {"NIDA": "hiv_addiction", "NIAAA": "hiv_addiction", "NIAID": "infectious_diseases",
                              "NLM": "llm_ai"},
    },
    "reporter": {
        "enabled": True, "schedule": "sun 04:15", "model_env": "LLM_MODEL", "window_days": 14,
        "max_candidates": 60, "batch_size": 15, "daily_cap_usd": 0.04,
        "award_types": ["1", "2"],             # domain search: new and competing awards
        "watch_award_types": ["1", "2", "3"],  # people_to_watch: also supplements; add "5" for annual continuations
    },
    "people": {
        "enabled": True, "schedule": "04:30", "model_env": "LLM_MODEL", "window_days": 14, "max_candidates": 40,
        "batch_size": 15, "daily_cap_usd": 0.02, "max_records": 300,
    },
    "briefing": {
        "enabled": True, "schedule": "04:45", "model_env": "LLM_MODEL_BRIEFING", "weekly_day": "monday",
        "min_items": 3, "max_items": 60, "lookback_hours": 26, "daily_cap_usd": 0.10, "min_relevance": 5,
    },
    "podcast": {
        "enabled": True, "on_demand": True, "separate_budget": True, "est_cost_usd": 0.22, "schedule": "on demand",
        "model_env": "LLM_MODEL_PODCAST", "target_minutes": 10, "monday_extra_minutes": 2,
        "words_per_minute": 175, "min_items": 3, "max_items": 30, "min_relevance": 5, "daily_cap_usd": 0.50, "keep_days": 30,
        "abstract_words": 90, "dive": True, "dive_minutes": 4, "dive_min_relevance": 6, "dive_attempts": 12, "fulltext_chars": 28000,
        "bitrate": "64k", "voice_dir": "data/voices", "lexicon": "config/pronunciation.yaml",
        "hosts": {"A": {"name": "Oliver", "model": "en_GB-alan-medium", "length_scale": 0.85},
                  "B": {"name": "Hannah", "model": "en_US-hfc_female-medium", "length_scale": 1.0}},
    },
    "tuning": {
        "enabled": True, "schedule": "05:00", "model_env": "LLM_MODEL_TUNING", "first_after": 10, "every": 20,
        "min_support": 3, "min_share": 0.7, "daily_cap_usd": 0.10,
    },
    "journal_club": {
        "enabled": True, "schedule": "mon 04:35", "model_env": "LLM_MODEL_JOURNAL_CLUB",
        "rotate_domains": True, "lookback_days": 7, "alternates": 2, "min_relevance": 6,
    },
}


def _is_v1(cfg: dict) -> bool:
    return cfg.get("version", 1) < 2


def migrate_v1_to_v2(v1: dict) -> dict:
    """Mechanical v1 -> v2 conversion. Literature behaviour is unchanged by construction."""
    for k in V1_KEYS:
        if k not in v1:
            raise ValueError(f"v1 config missing top-level key '{k}'")
    domains: dict[str, dict] = {d: {**DOMAIN_SEEDS[d], "literature_groups": []} for d in DOMAIN_ORDER}
    unmapped = []
    for g in v1["groups"]:
        doms = GROUP_DOMAINS.get(g["id"])
        if doms is None:
            doms = ["llm_ai"]
            unmapped.append(g["id"])
        home = doms[0]
        g2 = dict(g)
        if doms != [home]:
            g2["domains"] = doms
        domains[home]["literature_groups"].append(g2)
    if unmapped:
        log.warning("Groups not in the known mapping were placed under llm_ai; review: %s", unmapped)
    lim = v1["limits"]
    literature = {
        "enabled": True, "schedule": "04:00", "model_env": "LLM_MODEL",
        "sources": ["pubmed", "medrxiv", "arxiv"],
        "window_days": lim["window_days"], "max_candidates": lim["max_candidates"],
        "major_quota": lim["major_quota"], "batch_size": lim["batch_size"],
        "abstract_words": lim["abstract_words"], "feedback_examples": lim["feedback_examples"],
        "daily_cap_usd": None,
        "major": v1["major"], "medrxiv": v1["medrxiv"], "arxiv": v1["arxiv"],
        "exclude": v1["exclude"], "digest": v1["digest"],
    }
    streams = {"literature": literature, **{k: dict(v) for k, v in STREAM_DEFAULTS.items()}}
    return {"version": 2, "profile": v1["profile"], "people_to_watch": [], "learned": {"preferences": "", "prefer": [], "deprioritize": []},
            "domains": domains, "streams": streams}


def to_literature_cfg(v2: dict) -> dict:
    """v1-shaped view for the literature stream (queries, scoring, digest all consume this shape)."""
    lit = v2["streams"]["literature"]
    groups = []
    for dom_id, dom in v2["domains"].items():
        for g in dom.get("literature_groups", []):
            groups.append({**g, "domains": g.get("domains") or [dom_id]})
    return {
        "profile": v2["profile"], "groups": groups,
        "major": lit["major"], "medrxiv": lit["medrxiv"], "arxiv": lit["arxiv"], "exclude": lit["exclude"],
        "limits": {k: lit[k] for k in ("window_days", "max_candidates", "major_quota", "batch_size",
                                       "abstract_words", "feedback_examples")},
        "digest": lit["digest"], "domains": v2["domains"], "stream": lit,
    }


class _Dumper(yaml.SafeDumper):
    pass


def _str_repr(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_repr)

HEADER = """\
# Research-surveillance profile (version 2). Edit freely; re-read at every run.
#
# domains   : three research domains, each with weight, keywords, MeSH terms and (for the literature stream)
#             AND-structured PubMed query groups. Every item from every stream is tagged with 1+ domains by
#             keyword/MeSH matching, then confirmed/extended by the LLM in the same triage call.
#             `keywords` feed domain tagging and the grants / trials / awards searches; `*` is a wildcard and
#             short ALL-CAPS acronyms (HIV, LLM, AIDS) match case-sensitively.
# literature_groups : a group matches when EVERY clause in `all` matches (AND); inside a clause any `mesh`
#             heading or `tiab` phrase counts (OR). `weight` ranks candidates before the LLM sees them.
#             A group may list extra `domains:` it should tag.
# streams   : per-stream settings: enabled, schedule (informational; `lit gen-systemd` turns these into timers),
#             candidate caps, model env var, optional daily_cap_usd, and stream-specific options.
# people_to_watch : PIs whose new NIH awards are always surfaced, e.g.
#             - {name: "Last, First", institution: "optional"}
"""


def _dump(obj) -> str:
    return yaml.dump(obj, Dumper=_Dumper, sort_keys=False, width=118, allow_unicode=True,
                     default_flow_style=None).rstrip() + "\n"


def render_v2_yaml(v2: dict) -> str:
    out = [HEADER, "version: 2\n", "\n# What the LLM is told about you.\n", _dump({"profile": v2["profile"]}),
           "\n# New awards for these PIs are always shown (RePORTER stream).\n",
           _dump({"people_to_watch": v2["people_to_watch"]}),
           "\n# Soft preferences learned from your likes/dislikes. Maintained by approved tune-ups (🎛 Tuning on the page); edit freely.\n"
           "# They only re-rank (weights are clipped to -3..+2 on the 0-10 relevance scale); nothing is ever filtered out.\n",
           _dump({"learned": v2.get("learned") or {"preferences": "", "prefer": [], "deprioritize": []}}),
           "\n# ---- Domains ---------------------------------------------------------------------------\n",
           _dump({"domains": v2["domains"]}),
           "\n# ---- Streams ---------------------------------------------------------------------------\n",
           _dump({"streams": v2["streams"]})]
    return "".join(out)


def migrate_file(path: Path) -> str:
    """Rewrite a v1 interests.yaml as v2 (backup kept). Returns a unified diff of the change."""
    old_text = path.read_text()
    v1 = yaml.safe_load(old_text)
    if not _is_v1(v1):
        return ""
    backup = path.with_name(path.stem + ".v1.yaml.bak")
    if not backup.exists():
        shutil.copy(path, backup)
    new_text = render_v2_yaml(migrate_v1_to_v2(v1))
    path.write_text(new_text)
    return "".join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                        fromfile=backup.name, tofile=path.name, n=2))


def load_interests(path: Path) -> dict:
    """Load interests.yaml as v2. A v1 file still works (converted in memory; run `lit migrate-config`)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; copy config/interests.example.yaml to config/interests.yaml and edit the profile")
    cfg = yaml.safe_load(path.read_text())
    if _is_v1(cfg):
        log.info("%s is version 1; using an in-memory v2 conversion (run `lit migrate-config` to rewrite it)", path)
        cfg = migrate_v1_to_v2(cfg)
    for key in ("profile", "domains", "streams"):
        if key not in cfg:
            raise ValueError(f"{path}: missing top-level key '{key}'")
    cfg.setdefault("people_to_watch", [])
    from . import learned as _learned
    cfg["learned"] = _learned.normalise(cfg.get("learned"))
    return cfg


def save_people(path: Path, people: list[dict]) -> None:
    save_key(path, "people_to_watch", people)


def save_key(path: Path, key: str, value) -> None:
    """Replace one top-level key of interests.yaml. Atomic and lock-protected: the web server and the streams share this file."""
    import fcntl
    import os
    lock = path.with_name(".interests.lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cfg = yaml.safe_load(path.read_text())
        if _is_v1(cfg):
            cfg = migrate_v1_to_v2(cfg)
        cfg[key] = value
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(render_v2_yaml(cfg))
        os.replace(tmp, path)
