"""Guards against the page's HTML and script drifting apart (a stale cached index.html with a new app.js once blanked the page)."""
import re
from pathlib import Path

from litsearch.core import pagedata

WEB = Path(__file__).resolve().parent.parent / "web"
DYNAMIC = {"followbox", "wform", "wq", "wresults", "pod-audio", "pod-pick", "pod-ctl", "pod-ep", "pod-topic", "pod-custom"}          # created by the script itself (follow box), not in index.html


def ids_in_html():
    return set(re.findall(r'id="([\w-]+)"', (WEB / "index.html").read_text()))


def test_every_element_the_script_looks_up_exists_in_index_html():
    js = (WEB / "app.js").read_text()
    wanted = set(re.findall(r"\$\('#([\w-]+)'\)", js)) | set(re.findall(r"getElementById\('([\w-]+)'\)", js))
    missing = sorted(wanted - ids_in_html() - DYNAMIC)
    assert not missing, f"app.js looks up ids that index.html does not define: {missing}"


def test_stale_page_guard_lists_only_ids_that_exist():
    js = (WEB / "app.js").read_text()
    required = re.findall(r"'([\w-]+)'", re.search(r"const REQUIRED_IDS = \[(.*?)\];", js, re.S).group(1))
    assert required and not (set(required) - ids_in_html())                      # a typo here would reload-loop every visitor
    for must in ("brief-link", "person", "v-person", "toggles", "briefing"):    # the ids whose absence caused the original crash
        assert must in required


def test_asset_urls_are_versioned_and_stamped_with_a_content_hash(tmp_path):
    html = (WEB / "index.html").read_text()
    assert "app.js?v=__BUILD__" in html and "style.css?v=__BUILD__" in html
    (tmp_path / "web").mkdir()
    for n in pagedata.WEB_FILES:
        (tmp_path / "web" / n).write_text((WEB / n).read_text())
    b1 = pagedata.web_build(tmp_path / "web")
    (tmp_path / "web" / "app.js").write_text((WEB / "app.js").read_text() + "\n// changed")
    b2 = pagedata.web_build(tmp_path / "web")
    assert re.fullmatch(r"[0-9a-f]{10}", b1) and b1 != b2                          # any change to any file changes the version


def test_regenerate_writes_a_stamped_index_and_never_leaves_the_placeholder(tmp_path, db, cfg2):
    from datetime import datetime, timezone
    from litsearch.core import config
    (tmp_path / "web").mkdir()
    for n in pagedata.WEB_FILES:
        (tmp_path / "web" / n).write_text((WEB / n).read_text())
    s = config.Settings(env={}, root=tmp_path)
    pagedata.regenerate(s, cfg2, db, tmp_path / "site", datetime(2026, 11, 3, tzinfo=timezone.utc))
    out = (tmp_path / "site" / "index.html").read_text()
    build = pagedata.web_build(tmp_path / "web")
    assert "__BUILD__" not in out and f"app.js?v={build}" in out and f"style.css?v={build}" in out
    assert (tmp_path / "site" / "app.js").read_text() == (tmp_path / "web" / "app.js").read_text()


def test_new_features_are_gated_on_the_api_version_so_deploy_order_cannot_break_the_page():
    js = (WEB / "app.js").read_text()
    assert "apiV = ping.v || 1" in js and "apiV < 2" in js and "apiV < 3" in js   # tuning/reasons (2) and the podcast button (3) only against a server that has them
    from litsearch import web
    assert web.API_VERSION == 3
