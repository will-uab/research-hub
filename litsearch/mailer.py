"""SMTP delivery (HTML + plain-text alternative)."""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from .core.config import Settings

log = logging.getLogger(__name__)


def build_message(s: Settings, subject: str, html: str | None, text: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = s.get("MAIL_FROM")
    msg["To"] = s.get("MAIL_TO")
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    return msg


def send(s: Settings, subject: str, html: str | None, text: str) -> None:
    if not s.smtp_ready:
        raise RuntimeError("SMTP settings incomplete in .env (SMTP_HOST/USER/PASSWORD, MAIL_FROM, MAIL_TO)")
    msg = build_message(s, subject, html, text)
    host, port = s.get("SMTP_HOST"), int(s.get("SMTP_PORT", "587"))
    mode = s.get("SMTP_SECURITY", "starttls").lower()
    ctx = ssl.create_default_context()
    if mode == "ssl":
        server = smtplib.SMTP_SSL(host, port, timeout=60, context=ctx)
    else:
        server = smtplib.SMTP(host, port, timeout=60)
    with server:
        if mode == "starttls":
            server.starttls(context=ctx)
        server.login(s.get("SMTP_USER"), s.get("SMTP_PASSWORD"))
        server.send_message(msg)
    log.info("Sent email: %s", subject)
