"""NIH Guide for Grants and Contracts: policy / modification notices (NOFOs themselves are on Grants.gov now).

The RSS feed carries only the latest weekly batch, so the grants stream polls it daily.
"""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET

from ..core.http import RateLimiter, request
from .grantsgov import iso_date

log = logging.getLogger(__name__)
RSS = "https://grants.nih.gov/grants/guide/newsfeed/fundingopps.xml"
NOFO_RX = re.compile(r"\b(?:RFA|PAR|PAS|PA|OTA)-(?:[A-Z]{2}-)?\d{2}-\d{3}\b")
NOTICE_RX = re.compile(r"\b(NOT-[A-Z]{2}-\d{2}-\d{3})\b")


def nofo_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(NOFO_RX.findall(text or "")))


def parse_rss(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.findall("channel/item"):
        link = (it.findtext("link") or "").strip().replace("http://", "https://")
        blob = f"{link} {it.findtext('description') or ''} {it.findtext('title') or ''}"
        m = NOTICE_RX.search(blob)
        out.append({"number": m.group(1) if m else "", "title": (it.findtext("title") or "").strip(),
                    "link": link, "pub_date": (it.findtext("pubDate") or "").strip()})
    return out


def parse_notice(html_text: str) -> dict:
    """Extract what the grants stream needs from a Guide notice page."""
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", html_text, flags=re.S | re.I)
    title_m = re.search(r"<title>(.*?)</title>", t, re.S | re.I)
    text = html.unescape(re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", " ", t)))
    num = NOTICE_RX.search(text)
    exp = re.search(r"Expiration Date:\s*New Date\s*([A-Za-z]+ \d{1,2},? \d{4})\s*\(Prior Date\s*([A-Za-z]+ \d{1,2},? \d{4})", text)
    rel = re.search(r"Release Date:\s*([A-Za-z]+ \d{1,2},? \d{4})", text)
    fix = lambda s: iso_date(re.sub(r"^([A-Za-z]{3})[a-z]* (\d{1,2}),? (\d{4})", r"\1 \2, \3", s)) if s else ""
    return {"number": num.group(1) if num else "",
            "title": re.sub(r"\s+", " ", html.unescape(title_m.group(1))).strip() if title_m else "",
            "nofos": nofo_numbers(text), "release_date": fix(rel.group(1)) if rel else "",
            "expiration_new": fix(exp.group(1)) if exp else "", "expiration_prior": fix(exp.group(2)) if exp else ""}


def fetch_rss(limiter: RateLimiter | None = None) -> list[dict]:
    r = request("GET", RSS, limiter or RateLimiter(0.5), headers={"User-Agent": "Mozilla/5.0 litsearch"})
    return parse_rss(r.content)


def fetch_notice(link: str, limiter: RateLimiter | None = None) -> dict:
    r = request("GET", link, limiter or RateLimiter(0.5), headers={"User-Agent": "Mozilla/5.0 litsearch"})
    d = parse_notice(r.text)
    d["url"] = link
    return d
