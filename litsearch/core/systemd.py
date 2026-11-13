"""Generate one service + timer per enabled stream from the `schedule:` fields in interests.yaml."""
from __future__ import annotations

import re
from pathlib import Path

DAYS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}
TZ = "America/Chicago"


def on_calendar(schedule: str, tz: str = TZ) -> str:
    """'05:30' -> '*-*-* 05:30:00 TZ'; 'sun 06:15' -> 'Sun *-*-* 06:15:00 TZ'; 'mon-fri 04:55' -> 'Mon..Fri *-*-* 04:55:00 TZ'."""
    m = re.fullmatch(r"\s*(?:([A-Za-z]{3})[a-z]*(?:-([A-Za-z]{3})[a-z]*)?\s+)?(\d{1,2}):(\d{2})\s*", schedule or "")
    if not m:
        raise ValueError(f"bad schedule {schedule!r}: use 'HH:MM', 'ddd HH:MM' or 'ddd-ddd HH:MM' (e.g. 'sun 06:15', 'mon-fri 04:55')")
    day, day2, hh, mm = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"bad time in schedule {schedule!r}")
    if any(d and d.lower() not in DAYS for d in (day, day2)):
        raise ValueError(f"bad weekday in schedule {schedule!r}")
    prefix = (f"{DAYS[day.lower()]}..{DAYS[day2.lower()]} " if day2 else f"{DAYS[day.lower()]} ") if day else ""
    return f"{prefix}*-*-* {hh:02d}:{mm:02d}:00 {tz}"


def unit_name(stream: str) -> str:
    return "litsearch" if stream == "literature" else f"litsearch-{stream.replace('_', '-')}"   # literature keeps today's names


def web_unit(root: Path, user: str, port: int = 8081) -> str:
    """The page + API server. Loopback only; `tailscale serve` exposes it to the tailnet (that setup is unchanged)."""
    return f"""[Unit]
Description=Research hub web server: static page + JSON API (loopback only; exposed via tailscale serve)
After=network.target

[Service]
User={user}
WorkingDirectory={root}
ExecStart={root}/.venv/bin/python -m litsearch.web --bind 127.0.0.1 --port {port}
Restart=on-failure
NoNewPrivileges=true
ProtectSystem=strict
# The API writes ratings/saved marks (data/) and the watch list (config/), so the project folder is writable.
ReadWritePaths={root}
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"""


def render_units(cfg: dict, root: Path, user: str, tz: str = TZ) -> dict[str, str]:
    out: dict[str, str] = {"litsearch-web.service": web_unit(root, user)}
    for stream, scfg in cfg["streams"].items():
        if not scfg.get("enabled", True) or scfg.get("on_demand"):       # on-demand streams are started from the page, not a timer
            continue
        name = unit_name(stream)
        out[f"{name}.service"] = f"""[Unit]
Description=Research hub: {stream} stream
Wants=network-online.target
After=network-online.target time-sync.target

[Service]
Type=oneshot
User={user}
WorkingDirectory={root}
ExecStart={root}/.venv/bin/python -m litsearch --stream {stream}
# A hung network call should not block the next run.
TimeoutStartSec=30min
Nice=10
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths={root}
PrivateTmp=true
"""
        out[f"{name}.timer"] = f"""[Unit]
Description=Run the {stream} stream ({scfg['schedule']} {tz})

[Timer]
OnCalendar={on_calendar(scfg['schedule'], tz)}
# If the Pi was off or offline at the scheduled time, run once as soon as it is back.
Persistent=true
AccuracySec=1min
Unit={name}.service

[Install]
WantedBy=timers.target
"""
    return out


def install_commands(files: dict[str, str], root: Path) -> str:
    units = " ".join(f"{root}/systemd/{f}" for f in sorted(files))
    timers = " ".join(f for f in sorted(files) if f.endswith(".timer"))
    return (f"sudo cp {units} /etc/systemd/system/\nsudo systemctl daemon-reload\n"
            f"sudo systemctl enable --now {timers}\n"
            f"sudo systemctl enable litsearch-web.service && sudo systemctl restart litsearch-web.service   # picks up the new server\n"
            f"systemctl list-timers 'litsearch*' && systemctl status litsearch-web.service --no-pager | head -5")
