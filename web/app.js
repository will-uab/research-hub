(function () {
'use strict';
const $ = (s) => document.querySelector(s);
const setHidden = (sel, v) => { const e = $(sel); if (e) e.hidden = v; };
const ESC = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ESC[c]);
const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : '#');
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode etc. */ } },
  sget(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
  sset(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* ignore */ } },
};
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DEFAULTS = {view: 'domain', d: 'all', s: 'all', range: '14', rel: '0', q: '', sort: 'relevance', sv: '0', hr: '0', p: 'all'};
const STREAM_ORDER = ['literature', 'people', 'journal_club', 'grants', 'reporter', 'trials'];   // tab order
const SHORT = {literature: 'Literature', people: 'Person', journal_club: 'Journal club', grants: 'Grant', reporter: 'Award', trials: 'Trial'};
const PAGE = 60;

let S = {...DEFAULTS};
let items = [], status = null, fullLoaded = false, prev = null, shown = PAGE;
const shards = {};            // month -> {id: text}
let allAbstracts = false;
let watchList = [];
let briefing = {daily: null, weekly: null}, acked = new Set(), showAllBriefings = false, followOpen = false;
let tuningState = null, tuningOpen = false;
let podcast = {episodes: []}, podApi = null, podIdx = 0, podSig = '', podSpeed = 1;
let api = false, apiV = 0;                                   // true when the server with the JSON API is reachable
let marks = {ratings: {}, saved: new Set(), read: new Set()};

// ---------------------------------------------------------------- helpers
const fmtDate = (s) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s || '');
  return m ? `${MONTHS[+m[2] - 1]} ${+m[3]}${m[1] !== String(new Date().getFullYear()) ? ', ' + m[1] : ''}` : (s || '');
};
const ago = (iso) => {
  if (!iso) return 'never';
  const m = Math.round((Date.now() - Date.parse(iso)) / 60000);
  if (m < 2) return 'just now';
  if (m < 90) return m + ' min ago';
  if (m < 2880) return Math.round(m / 60) + ' h ago';
  return Math.round(m / 1440) + ' d ago';
};
const days = (iso) => {                      // whole days from today (server's date) to an ISO date; null if none
  if (!iso || !status) return null;
  return Math.round((new Date(iso + 'T00:00:00') - new Date(status.today + 'T00:00:00')) / 864e5);
};
const money = (n) => (n ? '$' + Math.round(n).toLocaleString('en-US') : 'n/a');
const usd = (n, d) => '$' + Number(n || 0).toFixed(d == null ? 3 : d);
const label = (id) => (status.streams.find((x) => x.id === id) || {label: id}).label;
const domLabel = (id) => (status.domains.find((x) => x.id === id) || {label: id}).label;
const badge = (txt, cls) => `<span class="badge ${cls || 'b-gray'}">${esc(txt)}</span>`;
const isSummary = (it) => it.ss && it.ss.kind === 'weekly_summary';

// ---------------------------------------------------------------- state: URL hash + localStorage
function parseHash() {
  const p = new URLSearchParams(location.hash.replace(/^#/, '')), o = {};
  Object.keys(DEFAULTS).forEach((k) => { if (p.has(k)) o[k] = p.get(k); });
  return o;
}
function loadState() {
  let saved = {};
  try { saved = JSON.parse(store.get('lh:state') || '{}'); } catch (e) { saved = {}; }
  S = {...DEFAULTS, ...saved, ...parseHash()};
  if (!['domain', 'stream', 'person'].includes(S.view)) S.view = 'domain';
  if (!['7', '14', '30', '90'].includes(String(S.range))) S.range = '14';
  S.range = String(S.range); S.rel = String(Math.max(0, Math.min(10, +S.rel || 0)));
  if (!['relevance', 'newest', 'deadline'].includes(S.sort)) S.sort = 'relevance';
}
function saveState() {
  const p = new URLSearchParams();
  Object.keys(S).forEach((k) => { if (S[k] !== DEFAULTS[k]) p.set(k, S[k]); });
  try { history.replaceState(null, '', p.toString() ? '#' + p.toString() : location.pathname + location.search); } catch (e) { /* ignore */ }
  store.set('lh:state', JSON.stringify(S));
}

// ---------------------------------------------------------------- data
async function getJSON(u) {
  const r = await fetch(u, {cache: 'no-cache'});
  if (!r.ok) throw new Error(u + ' -> HTTP ' + r.status);
  return r.json();
}
async function apiCall(method, path, body) {
  const opt = {method, headers: {'Content-Type': 'application/json', 'X-LH': '1'}, keepalive: true};
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || (path + ' -> HTTP ' + r.status));
  return data;
}
async function initApi() {
  try {
    const ping = await getJSON('api/ping');
    apiV = ping.v || 1;
    const st = await getJSON('api/state');
    api = true;
    marks = {ratings: st.ratings || {}, saved: new Set(st.saved || []), read: new Set(st.read || [])};
    acked = new Set(st.acked || []);
  } catch (e) { api = false; }                     // static-only hosting (e.g. a preview): buttons stay hidden
}
async function ensureFull() {
  if (fullLoaded) return;
  const d = await getJSON('data/items-90d.json');
  items = d.items; fullLoaded = true;
}
function loadShard(m) {
  if (!m) return Promise.resolve({});
  if (!shards[m]) shards[m] = getJSON(`data/abstracts/${m}.json`).catch(() => ({}));
  return shards[m];
}
async function abstractOf(it) { const m = await loadShard(it.shard); return m[String(it.id)] || ''; }
const textCache = {};
async function loadAllAbstracts() {
  if (allAbstracts) return;
  allAbstracts = true;
  const months = [...new Set(items.map((i) => i.shard).filter(Boolean))];
  const maps = await Promise.all(months.map(loadShard));
  months.forEach((m, k) => Object.keys(maps[k]).forEach((id) => { textCache[id] = maps[k][id]; }));
}

// ---------------------------------------------------------------- filtering
function haystack(it) {
  const ss = it.ss || {};
  const bits = [it.title, it.takeaway, it.fit_note, it.source_id];
  Object.keys(ss).forEach((k) => {
    const v = ss[k];
    if (typeof v === 'string') bits.push(v);
    else if (Array.isArray(v)) v.forEach((x) => bits.push(typeof x === 'string' ? x : (x && x.name) || ''));
  });
  if (textCache[it.id]) bits.push(textCache[it.id]);
  return bits.join(' ').toLowerCase();
}
function itemPeople(it) { const ss = it.ss || {}; return [...new Set([...(ss.watched || []), ...(ss.watched_authors || [])])]; }
function passes(it, cutoff, today, person) {
  const upcoming = it.stream === 'grants' && it.ss && it.ss.close_date && it.ss.close_date >= today;
  const saved = marks.saved.has(it.id);
  if (S.sv === '1' && !saved) return false;
  if (S.hr === '1' && marks.read.has(it.id)) return false;
  if (Date.parse(it.first_seen) < cutoff && !upcoming && !(S.sv === '1' && saved)) return false;     // saved items ignore the date range
  if (it.relevance != null && it.relevance < +S.rel) return false;
  if (S.view === 'person') {                          // only items about people you currently follow
    const who = person === undefined ? S.p : person, names = itemPeople(it).filter((n) => watchList.some((w) => w.name === n));
    if (!names.length || (who !== 'all' && !names.includes(who))) return false;
  } else if (S.d !== 'all' && !(it.domains || []).includes(S.d)) return false;
  if (S.s !== 'all' && it.stream !== S.s) return false;
  if (S.q.trim()) {
    const h = haystack(it);
    if (!S.q.toLowerCase().split(/\s+/).filter(Boolean).every((t) => h.includes(t))) return false;
  }
  return true;
}
function sortKey(it) { return it.relevance == null ? (isSummary(it) ? 99 : -1) : it.relevance + (it.adj || 0); }   // adj = learned preferences (ranking only)
function sorted(list) {
  const byNew = (a, b) => (a.first_seen < b.first_seen ? 1 : -1);
  if (S.sort === 'newest') return list.sort(byNew);
  if (S.sort === 'deadline') {
    const dl = (it) => { const d = it.stream === 'grants' && it.ss.close_date ? it.ss.close_date : null; return d && d >= status.today ? d : '9999'; };
    return list.sort((a, b) => (dl(a) < dl(b) ? -1 : dl(a) > dl(b) ? 1 : sortKey(b) - sortKey(a)));
  }
  return list.sort((a, b) => sortKey(b) - sortKey(a) || byNew(a, b));
}
function visible() {
  const cutoff = Date.now() - (+S.range) * 864e5;
  return sorted(items.filter((it) => passes(it, cutoff, status.today)));
}

// ---------------------------------------------------------------- cards
function evText(e) {
  const d = e.detail || {};
  const n = (s) => String(s || '').replace(/_/g, ' ').toLowerCase();
  if (e.kind === 'nofo_modified') {
    const ch = Object.keys(d.changes || {}).map((k) => `${k.replace(/_/g, ' ')}: ${d.changes[k][0] || '?'} → ${d.changes[k][1] || '?'}`);
    return `Modified (rev ${d.from_revision ?? '?'}→${d.to_revision ?? '?'})${ch.length ? ': ' + ch.join('; ') : ''}${d.comments ? ' — ' + d.comments : ''}`;
  }
  if (e.kind === 'notice') return `NIH notice ${d.notice}: ${d.title || ''}${d.expiration_new ? ` (expiration ${d.expiration_prior || '?'} → ${d.expiration_new})` : ''}`;
  if (e.kind === 'reissued') return `Reissued as ${d.new_number}${d.new_title ? ': ' + d.new_title : ''}`;
  if (e.kind === 'status_change') return `Status: ${n(d.from)} → ${n(d.to)}${d.why_stopped ? ' (' + d.why_stopped + ')' : ''}`;
  if (e.kind === 'results_posted') return `Results posted${d.date ? ' ' + fmtDate(d.date) : ''}`;
  return e.kind;
}
function eventsHtml(it) {
  const cut = Date.now() - 30 * 864e5;
  const ev = (it.events || []).filter((e) => Date.parse(e.ts) >= cut);
  return ev.map((e) => `<div class="ev">${esc(evText(e))}${e.kind === 'notice' && e.detail && e.detail.url ? ` · <a href="${esc(safeUrl(e.detail.url))}">notice</a>` : ''}</div>`).join('');
}
function isNew(it) { return Date.parse(it.first_seen) > prevMs(); }
function wasUpdated(it) { return (it.events || []).some((e) => Date.parse(e.ts) > prevMs()); }
function prevMs() { return prev ? Date.parse(prev) : Date.now() - 864e5; }
function domPills(it) { return (it.domains || []).map((d) => `<span class="pill">${esc(domLabel(d))}</span>`).join(' '); }
function absBlock(it, title) {
  return `<details data-abs="${it.id}"><summary>${esc(title || 'Abstract')}</summary><p>Loading…</p></details>`;
}
function relBit(it) { return it.relevance != null ? ` · relevance ${Math.round(it.relevance)}/10` : ' · <span class="note">not LLM-scored</span>'; }
function deadlineBadge(it) {
  const c = it.ss.close_date, dd = days(c), flag = status.deadline_flag_days || 60;
  if (dd == null) return '';
  if (dd < 0) return badge('CLOSED ' + fmtDate(c), 'b-gray');
  if (dd <= 30) return badge(`DUE IN ${dd}d`, 'b-red');
  if (dd <= flag) return badge(`DUE IN ${dd}d`, 'b-amber');
  return badge('due ' + fmtDate(c), 'b-gray');
}
function kv(pairs) {
  return '<div class="kv">' + pairs.filter((p) => p[1] !== '' && p[1] != null).map((p) => `<span>${esc(p[0])} <b>${esc(p[1])}</b></span>`).join('') + '</div>';
}
function head(it, badges) {
  const tag = S.view === 'domain' && S.s === 'all' ? '' : '';
  return `<div class="t">${badges}<a href="${esc(safeUrl(it.url))}">${esc(it.title)}</a></div>`;
}
function newBadges(it) { return (isNew(it) ? badge('NEW', 'b-new') : '') + (wasUpdated(it) ? badge('UPDATED', 'b-amber') : ''); }

function cardLiterature(it) {
  const ss = it.ss || {};
  const who = ss.watched || ss.watched_authors || [];
  const b = (who.length ? badge('👤 FOLLOWING', 'b-red') : '') + (ss.is_preprint ? badge('PREPRINT', 'b-amber') : '')
    + (ss.is_major || it.track === 'major' ? badge('MAJOR', 'b-lit') : '') + (it.status === 'notice' ? badge('NOTICE', 'b-red') : '') + newBadges(it);
  const links = (ss.links || []).map((l) => `<a href="${esc(safeUrl(l[1]))}">${esc(l[0])}</a>`).join(' · ');
  return `<article class="card">${head(it, b)}
    <div class="meta">${esc(ss.authors || '')}${ss.authors ? ' · ' : ''}<em>${esc(ss.journal || '')}</em> · ${esc(fmtDate(it.date))}${relBit(it)}</div>
    ${who.length ? `<div class="meta">Following: ${esc(who.join('; '))}</div>` : ''}
    <div>${domPills(it)}</div>
    ${it.takeaway ? `<div class="take">${esc(it.takeaway)}</div>` : ''}${it.fit_note ? `<div class="why">Why: ${esc(it.fit_note)}</div>` : ''}
    ${absBlock(it)}
    <div class="foot2">${links}${!api && it.stream === 'literature' ? `${links ? ' · ' : ''}feedback: <code>lit feedback +${esc(it.source_id)}</code> / <code>-${esc(it.source_id)}</code>` : ''}</div></article>`;
}
function cardGrant(it) {
  const ss = it.ss || {};
  if (isSummary(it)) return cardGrantSummary(it);
  const b = deadlineBadge(it) + (ss.status === 'forecasted' ? badge('FORECAST', 'b-lit') : '') + newBadges(it)
    + (ss.pref_institute || ss.pref_mechanism ? badge('PREFERRED', 'b-green') : '');
  return `<article class="card">${head(it, b)}
    ${kv([['NOFO', ss.number], ['Institute', (ss.institutes || []).join(', ')], ['Mechanism', ss.mechanism], ['Opens', fmtDate(ss.open_date)],
      ['Due', ss.close_date ? fmtDate(ss.close_date) : ''], ['Expires', ss.archive_date ? fmtDate(ss.archive_date) : ''],
      ['Ceiling', ss.award_ceiling ? money(+ss.award_ceiling) : '']])}
    <div>${domPills(it)}${relBit(it)}</div>
    ${it.takeaway ? `<div class="take">${esc(it.takeaway)}</div>` : ''}${it.fit_note ? `<div class="why">Fit: ${esc(it.fit_note)}</div>` : ''}
    ${eventsHtml(it)}${absBlock(it, 'Synopsis')}</article>`;
}
function cardGrantSummary(it) {
  const ss = it.ss;
  const li = (v) => `<li><a href="${esc(safeUrl(v.url))}">${esc(v.title)}</a> <span class="meta">${esc(v.number || '')}${v.close_date ? ' · due ' + esc(fmtDate(v.close_date)) : ''}</span></li>`;
  return `<article class="card"><div class="t">${badge('WEEKLY', 'b-lit')}${esc(it.title)}</div>
    <div class="meta">${(ss.new || []).length} new relevant · ${(ss.closing || []).length} due within ${ss.flag_days || 60} days · ${ss.modified || 0} modification/notice event(s)</div>
    ${(ss.new || []).length ? `<div class="wn-head">New this week</div><ul class="wn-list">${ss.new.slice(0, 8).map(li).join('')}</ul>` : ''}
    ${(ss.closing || []).length ? `<div class="wn-head">Closing soon</div><ul class="wn-list">${ss.closing.slice(0, 8).map(li).join('')}</ul>` : ''}</article>`;
}
function watchButtons(ss) {
  const have = new Set(watchList.map((p) => String(p.profile_id)));
  const todo = (ss.pi_profiles || []).filter((p) => !have.has(String(p.profile_id)));
  return todo.length ? `<div class="kv">${todo.map((p) => `<button class="pill" data-watch-pi="${esc(p.profile_id)}" data-name="${esc(p.name)}">＋ Follow ${esc(p.name)}</button>`).join(' ')}</div>` : '';
}
function cardAward(it) {
  const ss = it.ss || {};
  const w = (ss.watched || []).length;
  const b = (w ? badge('👤 FOLLOWING', 'b-red') : '') + newBadges(it);
  const others = (ss.pis || []).length > 1 ? ` +${ss.pis.length - 1} co-PI` : '';
  return `<article class="card">${head(it, b)}
    ${kv([['PI', (ss.contact_pi || '') + others], ['Institution', ss.institution], ['Institute', ss.institute], ['Mechanism', ss.mechanism],
      ['FY', ss.fiscal_year], ['Award', money(ss.amount)], ['Notice', fmtDate(ss.award_notice_date)], ['Project', ss.project_num]])}
    ${w ? `<div class="meta">Following: ${esc(ss.watched.join('; '))}</div>` : ''}
    ${api ? watchButtons(ss) : ''}
    <div>${domPills(it)}${relBit(it)}</div>
    ${it.takeaway ? `<div class="take">${esc(it.takeaway)}</div>` : ''}${it.fit_note ? `<div class="why">Why: ${esc(it.fit_note)}</div>` : ''}
    ${absBlock(it)}</article>`;
}
const STATUS_CLS = {RECRUITING: 'b-green', ENROLLING_BY_INVITATION: 'b-green', NOT_YET_RECRUITING: 'b-lit', ACTIVE_NOT_RECRUITING: 'b-gray',
  COMPLETED: 'b-gray', TERMINATED: 'b-red', WITHDRAWN: 'b-red', SUSPENDED: 'b-amber'};
function cardTrial(it) {
  const ss = it.ss || {};
  const b = badge(String(ss.status || '').replace(/_/g, ' '), STATUS_CLS[ss.status] || 'b-gray') + (ss.has_results ? badge('RESULTS POSTED', 'b-green') : '')
    + (ss.ai_intervention ? badge('AI', 'b-lit') : '') + newBadges(it);
  const enr = ss.enrollment ? `${ss.enrollment}${ss.enrollment_type ? ' (' + ss.enrollment_type + ')' : ''}` : '';
  return `<article class="card">${head(it, b)}
    ${kv([['NCT', ss.nct_id], ['Sponsor', ss.sponsor], ['Phase', (ss.phases || []).join(', ')], ['Enrollment', enr], ['Registered', fmtDate(ss.first_post)]])}
    ${kv([['Intervention', (ss.interventions || []).map((i) => i.name).slice(0, 3).join('; ')], ['Primary outcome', (ss.primary_outcomes || [])[0]]])}
    <div>${domPills(it)}${relBit(it)}</div>
    ${it.takeaway ? `<div class="take">${esc(it.takeaway)}</div>` : ''}${it.fit_note ? `<div class="why">Why: ${esc(it.fit_note)}</div>` : ''}
    ${eventsHtml(it)}${absBlock(it, 'Summary')}</article>`;
}
function guideHtml(ss) {
  if (!ss.generated && !ss.why_discuss) return '<div class="note">Discussion guide not generated (budget guard, or a model error). The abstract is below.</div>';
  return `<div class="guide"><h4>Why discuss</h4>${esc(ss.why_discuss)}
    <h4>Discussion questions</h4><ol>${(ss.questions || []).map((q) => `<li>${esc(q)}</li>`).join('')}</ol>
    <h4>Methods teaching point</h4>${esc(ss.methods_point)}
    ${(ss.background_reading || []).length ? `<h4>Suggested background</h4><ul>${ss.background_reading.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}</div>`;
}
function cardPick(it, featured) {
  const ss = it.ss || {};
  const b = badge(ss.role === 'primary' ? 'PRIMARY PICK' : 'ALTERNATE', ss.role === 'primary' ? 'b-lit' : 'b-gray') + badge(ss.week || '', 'b-gray')
    + (ss.is_preprint ? badge('PREPRINT', 'b-amber') : '');
  return `<article class="${featured ? 'featured' : 'card'}">${head(it, b)}
    <div class="meta">${esc(ss.authors || '')}${ss.authors ? ' · ' : ''}<em>${esc(ss.journal || '')}</em>${ss.pick_domain ? ' · this week: ' + esc(domLabel(ss.pick_domain)) : ''}</div>
    ${it.takeaway ? `<div class="take">${esc(it.takeaway)}</div>` : ''}${guideHtml(ss)}
    <div class="note">Based on the abstract only; read the paper before the meeting.</div>${absBlock(it)}</article>`;
}
function cardPickCompact(it) {          // What's-new version: guide collapsed so the panel stays short on a phone
  const ss = it.ss || {};
  return `<article class="featured"><div class="t">${badge('PRIMARY PICK', 'b-lit')}${badge(ss.week || '', 'b-gray')}<a href="${esc(safeUrl(it.url))}">${esc(it.title)}</a></div>
    <div class="meta">${esc(ss.authors || '')}${ss.authors ? ' · ' : ''}<em>${esc(ss.journal || '')}</em>${ss.pick_domain ? ' · this week: ' + esc(domLabel(ss.pick_domain)) : ''}</div>
    <details><summary>Discussion guide</summary>${guideHtml(ss)}<div class="note">Based on the abstract only.</div></details>
    <div class="foot2"><a href="#" data-go="journal_club">All picks (primary + alternates)</a></div></article>`;
}
function actsHtml(it) {
  if (!api) return '';
  const r = marks.ratings[it.id];
  return `<div class="acts"><button class="act ${r === 1 ? 'on' : ''}" data-act="up" aria-label="More like this" title="More like this">👍</button>
    <button class="act ${r === -1 ? 'on' : ''}" data-act="down" aria-label="Less like this" title="Less like this">👎</button>
    <button class="act ${marks.saved.has(it.id) ? 'on' : ''}" data-act="save" aria-label="Save" title="Save for later">🔖</button></div>`;
}
function stateClasses(id) {
  return (marks.read.has(id) ? ' read' : '') + (marks.ratings[id] === -1 ? ' disliked' : '') + (marks.saved.has(id) ? ' saved' : '');
}
function adjTag(it) {                              // why an item sits higher/lower than its raw score: your approved preferences
  return it.adj_why && it.adj_why.length ? `<div class="adjtag" title="From your learned preferences (ranking only; nothing is hidden)">${it.adj_why.map(esc).join(' · ')}</div>` : '';
}
function withActions(it, html) {
  return html.replace(/^<article class="([^"]*)"/, (m, c) => `<article class="${c}${stateClasses(it.id)}" data-id="${it.id}"`)
    .replace(/<\/article>\s*$/, adjTag(it) + actsHtml(it) + '</article>');
}
function card(it) { return withActions(it, cardInner(it)); }
function cardInner(it) {
  switch (it.stream) {
    case 'literature': return cardLiterature(it);
    case 'people': return cardLiterature(it);
    case 'grants': return cardGrant(it);
    case 'reporter': return cardAward(it);
    case 'trials': return cardTrial(it);
    case 'journal_club': return cardPick(it, false);
    default: return `<article class="card">${head(it, '')}</article>`;
  }
}

// ---------------------------------------------------------------- status banner + what's new
function renderStatus() {
  const st = status, sp = st.spend, enabled = st.streams.filter((s) => s.enabled);
  const failed = enabled.filter((s) => s.last_run_ok === false);
  const stale = enabled.filter((s) => s.stale && !s.never_run && s.last_run_ok !== false);
  const never = enabled.filter((s) => s.never_run);
  const notes = enabled.filter((s) => (s.notes || []).length);
  const cls = failed.length ? 'red' : (stale.length || notes.length) ? 'amber' : never.length ? 'gray' : 'green';
  const txt = failed.length ? `${failed.map((s) => s.label).join(', ')} failed`
    : stale.length ? `${stale.map((s) => s.label).join(', ')} overdue`
    : notes.length ? 'Notes on some streams'
    : never.length === enabled.length ? 'No runs yet'
    : never.length ? `Waiting for first run: ${never.map((s) => s.label).join(', ')}` : 'All streams OK';
  const podSep = (sp.separate || {}).podcast;
  const pct = Math.min(100, 100 * sp.today / sp.daily_cap), bcls = sp.today >= sp.daily_cap ? 'over' : pct >= 80 ? 'warn' : '';
  const wk = sp.week_journal_club, wpct = Math.min(100, 100 * wk / sp.weekly_extra), wcls = wk >= sp.weekly_extra ? 'over' : wpct >= 80 ? 'warn' : '';
  const rows = enabled.map((s) => `<tr><td><b>${esc(s.label)}</b><br><span class="note">${esc(s.schedule)}</span></td>
    <td>${s.last_ok ? esc(ago(s.last_ok)) : '<span class="note">never</span>'}${s.stale && !s.never_run ? ' <span class="badge b-amber">overdue</span>' : ''}</td>
    <td>${s.never_run ? '<span class="note">not run yet</span>' : s.last_run_ok ? 'OK' : '<span class="badge b-red">FAILED</span> ' + esc(s.error)}
      ${(s.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join('')}${s.cap ? `<div class="note">cap ${usd(s.cap, 2)}/day</div>` : ''}</td>
    <td>${usd(s.spend_today)}</td></tr>`).join('');
  const byDay = {}; sp.log.forEach((r) => { byDay[r.day] = (byDay[r.day] || 0) + r.cost; });
  const logTxt = Object.keys(byDay).sort().reverse().slice(0, 7).map((d) => `${fmtDate(d)} ${usd(byDay[d])}`).join(' · ');
  $('#status').innerHTML = `<details><summary><span class="dot ${cls}"></span><b>${esc(txt)}</b>
    <span>· API ${usd(sp.today)} / ${usd(sp.daily_cap, 2)} today</span><span>· Journal club ${usd(wk)} / ${usd(sp.weekly_extra, 2)} this week</span>${podSep ? `<span>· Podcast ${usd(podSep.today)} / ${usd(podSep.cap, 2)} today (separate)</span>` : ''}</summary>
    <div class="body"><div class="bar ${bcls}"><i style="width:${pct}%"></i></div><div class="bar ${wcls}"><i style="width:${wpct}%"></i></div>
    <table><tr><th>Stream</th><th>Last success</th><th>Last run</th><th>Today</th></tr>${rows}</table>
    ${logTxt ? `<div>Spend log: ${esc(logTxt)}</div>` : ''}<div class="note">Combined this week: ${usd(sp.week)} · data generated ${esc(ago(st.generated))}</div></div></details>`;
}
function setView(view, d, s, extra) {
  S.view = view; if (d != null) S.d = d; if (s != null) S.s = s; Object.assign(S, extra || {}); shown = PAGE; update(true);
  window.scrollTo({top: $('.controls').offsetTop - 4, behavior: 'smooth'});
}
// Featured: the few items most likely to matter, scored in the browser from fields already in the data (no LLM call).
const FEATURE_PER_STREAM = 2, FEATURE_N = 5;
function featureScore(it) {
  const ss = it.ss || {};
  let s = (it.relevance != null ? it.relevance : 3) + (it.adj || 0);        // un-scored (budget fallback) items rank low; adj = learned preferences
  const w = Math.max(0, ...(it.domains || []).map((d) => (status.domains.find((x) => x.id === d) || {}).weight || 0));
  s += (w / 10) * 2;                                                         // LLM/AI (weight 10) counts most
  if (isNew(it)) s += 1;
  if (wasUpdated(it)) s += 0.5;
  if (it.stream === 'reporter' && (ss.watched || []).length) s += 20;       // watched PIs always surface
  if (it.stream === 'journal_club' && ss.role === 'primary') s += 3;
  if (it.stream === 'trials') s += (ss.ai_intervention ? 1 : 0) + (ss.has_results ? 1.5 : 0);
  if (it.stream === 'grants') s += (ss.pref_institute ? 1 : 0) + (ss.pref_mechanism ? 0.5 : 0);
  if (it.stream === 'literature' && (ss.is_major || it.track === 'major')) s += 1;
  return s;
}
function featureReasons(it) {
  const ss = it.ss || {}, r = [];
  if (it.stream === 'reporter' && (ss.watched || []).length) r.push('👤 following: ' + ss.watched.join('; '));
  if (it.stream === 'journal_club') r.push('journal club pick');
  if (it.relevance != null) r.push(`relevance ${Math.round(it.relevance)}/10`);
  if (ss.has_results) r.push('results posted');
  if (ss.ai_intervention) r.push('AI intervention');
  if (it.stream === 'grants' && (ss.pref_institute || ss.pref_mechanism)) r.push('preferred ' + [ss.pref_institute ? 'institute' : '', ss.pref_mechanism ? 'mechanism' : ''].filter(Boolean).join(' + '));
  if (it.stream === 'literature' && (ss.is_major || it.track === 'major')) r.push('major journal');
  if ((it.domains || []).length) r.push((it.domains || []).map(domLabel).join(' / '));
  return r.slice(0, 4);
}
function pickFeatured(since) {
  const pickPapers = new Set(items.filter((i) => i.stream === 'journal_club' && i.ss.role === 'primary').map((i) => i.ss.paper_item_id));
  const eligible = items.filter((i) => !isSummary(i) && i.status !== 'notice' && marks.ratings[i.id] !== -1 && !(i.stream === 'journal_club' && i.ss.role !== 'primary')
    && !(i.stream === 'literature' && pickPapers.has(i.id)));            // the journal club pick stands in for its source paper
  const rank = (list) => list.map((i) => [featureScore(i), i]).sort((a, b) => b[0] - a[0] || (a[1].first_seen < b[1].first_seen ? 1 : -1)).map((x) => x[1]);
  const taken = {}, out = [];
  const add = (list) => rank(list).forEach((i) => {
    if (out.length >= FEATURE_N || out.includes(i) || (taken[i.stream] || 0) >= FEATURE_PER_STREAM) return;
    taken[i.stream] = (taken[i.stream] || 0) + 1; out.push(i);
  });
  add(eligible.filter((i) => Date.parse(i.first_seen) > since));                         // new since your last visit first
  if (out.length < FEATURE_N) add(eligible.filter((i) => Date.parse(i.first_seen) > Date.now() - 7 * 864e5));   // then the past week
  return out;
}
function renderWhatsNew() {
  const since = prevMs();
  const fresh = items.filter((i) => Date.parse(i.first_seen) > since && !isSummary(i));
  const counts = {}; fresh.forEach((i) => { counts[i.stream] = (counts[i.stream] || 0) + 1; });
  const feat = pickFeatured(since);
  const newGrants = items.filter((i) => i.stream === 'grants' && i.ss && i.ss.kind === 'nofo' && Date.parse(i.first_seen) > since)
    .sort((a, b) => (b.relevance || 0) - (a.relevance || 0));
  const watched = items.filter((i) => i.stream === 'reporter' && (i.ss.watched || []).length && Date.parse(i.first_seen) > since && !feat.includes(i));
  const results = items.filter((i) => i.stream === 'trials' && !feat.includes(i) && ((i.events || []).some((e) => e.kind === 'results_posted' && Date.parse(e.ts) > since)
    || (Date.parse(i.first_seen) > since && i.ss.has_results)));
  const sinceTxt = prev ? 'since your last visit (' + ago(prev) + ')' : 'in the last 24 hours';
  const pills = STREAM_ORDER.filter((s) => counts[s]).map((s) => `<button class="pill" data-go="${s}">${esc(label(s))} ${counts[s]}</button>`).join('')
    || '<span class="note">Nothing new yet.</span>';
  const featHtml = feat.map((i) => {
    const take = i.takeaway || i.fit_note || '';
    return `<li><span class="pill">${esc(SHORT[i.stream] || i.stream)}</span> <a href="${esc(safeUrl(i.url))}">${esc(i.title)}</a>
      <div class="reason">${esc(featureReasons(i).join(' · '))}</div>${take ? `<div class="meta">${esc(take)}</div>` : ''}</li>`;
  }).join('');
  const grantLi = (i) => `<li><a href="${esc(safeUrl(i.url))}">${esc(i.title)}</a> <span class="meta">${esc(i.ss.number || '')}${(i.ss.institutes || []).length ? ' · ' + esc(i.ss.institutes.join(', ')) : ''}${i.ss.mechanism ? ' · ' + esc(i.ss.mechanism) : ''}</span></li>`;
  $('#whatsnew').innerHTML = `<details open><summary><h2 style="display:inline">What's new</h2> <span class="sub">${esc(sinceTxt)} · ${fresh.length} new</span></summary>
    <div class="wn-counts">${pills}</div>
    ${feat.length ? `<div class="wn-head">★ Featured · most likely to interest you</div><ol class="feat">${featHtml}</ol>` : ''}
    <div class="wn-head">New grant announcements${newGrants.length ? ' · ' + newGrants.length : ''}</div>
    ${newGrants.length ? `<ul class="wn-list">${newGrants.slice(0, 4).map(grantLi).join('')}</ul>
      ${newGrants.length > 4 ? `<button class="pill" data-go="grants">All ${newGrants.length} new grants</button>` : ''}`
      : '<div class="note">No new grant announcements since then.</div>'}
    ${watched.length ? `<div class="wn-head">👤 New awards for people you follow</div><ul class="wn-list">${watched.slice(0, 3).map((i) => `<li><a href="${esc(safeUrl(i.url))}">${esc(i.title)}</a> <span class="meta">${esc(i.ss.watched.join('; '))}</span></li>`).join('')}</ul>` : ''}
    ${results.length ? `<div class="wn-head">📊 Trials with newly posted results</div><ul class="wn-list">${results.slice(0, 3).map((i) => `<li><a href="${esc(safeUrl(i.url))}">${esc(i.title)}</a> <span class="meta">${esc(i.ss.nct_id)}</span></li>`).join('')}</ul>` : ''}
    </details>`;
}

// ---------------------------------------------------------------- controls + list
function tabs(el, entries, cur, attr) {
  el.innerHTML = entries.map(([id, lab]) => `<button role="tab" aria-selected="${id === cur}" class="${id === cur ? 'on' : ''}" data-${attr}="${esc(id)}">${esc(lab)}</button>`).join('');
}
function renderControls() {
  $('#v-domain').setAttribute('aria-selected', S.view === 'domain'); $('#v-stream').setAttribute('aria-selected', S.view === 'stream');
  const domTabs = status.domains.map((d) => [d.id, d.label]).concat([['all', 'All']]);
  const strTabs = STREAM_ORDER.map((s) => [s, label(s)]).concat([['all', 'All']]);
  const domChips = [['all', 'All domains']].concat(status.domains.map((d) => [d.id, d.label]));
  const strChips = [['all', 'All streams']].concat(STREAM_ORDER.map((s) => [s, label(s)]));
  const vp = $('#v-person'); if (vp) vp.setAttribute('aria-selected', S.view === 'person');
  if (S.view === 'domain') { tabs($('#primary'), domTabs, S.d, 'd'); tabs($('#secondary'), strChips, S.s, 's'); }
  else if (S.view === 'person') { renderPersonTabs(); tabs($('#secondary'), strChips.filter(([id]) => id === 'all' || PERSON_STREAMS.includes(id)), S.s, 's'); }
  else { tabs($('#primary'), strTabs, S.s, 's'); tabs($('#secondary'), domChips, S.d, 'd'); }
  setHidden('#toggles', !api); const osv = $('#onlysaved'), hrd = $('#hideread'); if (osv) osv.checked = S.sv === '1'; if (hrd) hrd.checked = S.hr === '1';
  $('#range').value = S.range; $('#sort').value = S.sort; $('#rel').value = S.rel; $('#relval').textContent = S.rel; $('#q').value = S.q;
}
function renderList() {
  const list = visible();
  const byStream = S.view === 'domain' && S.s === 'all';
  let html = '';
  if (byStream) {
    let n = 0;
    STREAM_ORDER.forEach((sid) => {
      const g = list.filter((i) => i.stream === sid);
      if (!g.length || n >= shown) return;
      const take = g.slice(0, Math.max(0, shown - n)); n += take.length;
      html += `<div class="group-h">${esc(label(sid))} · ${g.length}</div>` + take.map(card).join('');
    });
  } else html = list.slice(0, shown).map(card).join('');
  $('#list').innerHTML = html || '<div class="empty">Nothing matches these filters. Try a longer range or a lower minimum relevance.</div>';
  $('#count').textContent = `${list.length} item${list.length === 1 ? '' : 's'}`;
  $('#more').hidden = list.length <= shown;
  $('#more').textContent = `Show more (${list.length - shown} left)`;
}
function renderFoot() {
  $('#foot').textContent = `${items.length} items loaded${fullLoaded ? ' (90-day window)' : ' (recent; longer ranges load more)'} · generated ${ago(status.generated)}`;
}
async function update(persist) {
  if ((+S.range > 14 || S.sv === '1') && !fullLoaded) { try { await ensureFull(); } catch (e) { showErr(e); } }
  if (S.q.trim().length >= 3 && !allAbstracts) { loadAllAbstracts().then(() => renderList()).catch(showErr); }
  renderControls(); renderPerson(); renderList(); renderFoot();
  if (persist) saveState();
}
function showErr(e) { const el = $('#err'); el.hidden = false; el.textContent = 'Error: ' + (e && e.message ? e.message : e); }

// ---------------------------------------------------------------- events
const REASONS = {
  '-1': [['not_topic', 'Not my topic'], ['already_knew', 'Already knew it'], ['weak_study', 'Weak study'], ['wrong_population', 'Wrong population'], ['too_basic', 'Too basic']],
  '1': [['on_topic', 'On topic'], ['great_method', 'Great method'], ['useful_project', 'Useful for a project'], ['follow_person', 'Follow this person']]};
const reasonTimers = {};
function clearReasons(art) { const r = art.querySelector('.reasons'); if (r) r.remove(); clearTimeout(reasonTimers[art.dataset.id]); }
function showReasons(art, id, sign) {                 // optional one-tap "why?" after a rating; fades away on its own
  clearReasons(art);
  if (apiV < 2) return;                               // older server has no reason endpoint
  const row = document.createElement('div');
  row.className = 'reasons';
  row.innerHTML = '<span class="note">Why? (optional)</span>' + REASONS[String(sign)].map(([k, l]) => `<button class="pill" data-reason="${k}" data-id="${id}">${l}</button>`).join('') + '<button class="x" data-reason="" aria-label="Skip">✕</button>';
  const acts = art.querySelector('.acts'); if (acts) acts.after(row); else art.appendChild(row);
  reasonTimers[id] = setTimeout(() => row.remove(), 9000);
}
async function sendReason(btn) {
  const id = +btn.dataset.id, reason = btn.dataset.reason, art = btn.closest('[data-id]');
  if (!reason) { clearReasons(art); return; }
  btn.classList.add('on');
  try { await apiCall('POST', 'api/reason', {id, reason}); btn.textContent = '✓ ' + btn.textContent; setTimeout(() => clearReasons(art), 900); }
  catch (e) { btn.classList.remove('on'); showErr(e); }
}
function refreshCard(id) {
  document.querySelectorAll(`[data-id="${id}"]`).forEach((el) => {
    el.classList.toggle('read', marks.read.has(id));
    el.classList.toggle('disliked', marks.ratings[id] === -1);
    el.classList.toggle('saved', marks.saved.has(id));
    const on = {up: marks.ratings[id] === 1, down: marks.ratings[id] === -1, save: marks.saved.has(id)};
    el.querySelectorAll('[data-act]').forEach((b) => b.classList.toggle('on', !!on[b.dataset.act]));
  });
}
async function handleAct(btn) {
  const art = btn.closest('[data-id]'), id = +art.dataset.id, act = btn.dataset.act;
  const prevRating = marks.ratings[id] || 0, prevSaved = marks.saved.has(id);
  try {
    if (act === 'save') {
      if (prevSaved) marks.saved.delete(id); else marks.saved.add(id);
      refreshCard(id);
      await apiCall('POST', 'api/mark', {id, saved: !prevSaved});
    } else {
      const want = act === 'up' ? 1 : -1, next = prevRating === want ? 0 : want;       // tap again to clear
      if (next) marks.ratings[id] = next; else delete marks.ratings[id];
      refreshCard(id);
      await apiCall('POST', 'api/rate', {id, sign: next});
      if (next) showReasons(art, id, next); else clearReasons(art);
    }
  } catch (e) {                                                                         // roll back the optimistic change
    if (act === 'save') { if (prevSaved) marks.saved.add(id); else marks.saved.delete(id); }
    else if (prevRating) marks.ratings[id] = prevRating; else delete marks.ratings[id];
    refreshCard(id); showErr(e);
  }
}
document.addEventListener('click', (ev) => {
  const rs = ev.target.closest('[data-reason]');
  if (rs) { sendReason(rs); return; }
  const act = ev.target.closest('[data-act]');
  if (act) { handleAct(act); return; }
  const wp = ev.target.closest('[data-watch-pi]');
  if (wp) { watchPi(wp); return; }
  const link = ev.target.closest('.t a, .take a');                                      // opening an item marks it read
  if (link && api) {
    const art = link.closest('[data-id]');
    if (art && !marks.read.has(+art.dataset.id)) { const id = +art.dataset.id; marks.read.add(id); refreshCard(id); apiCall('POST', 'api/mark', {id, read: true}).catch(() => {}); }
  }
});
document.addEventListener('click', (ev) => {
  const t = ev.target.closest('[data-view],[data-d],[data-s],[data-p],[data-go],#more');
  if (!t) return;
  if (t.tagName === 'A') ev.preventDefault();
  if (t.id === 'more') { shown += PAGE; renderList(); return; }
  if (t.dataset.view) { S.view = t.dataset.view; }
  else if (t.dataset.d != null) { S.d = t.dataset.d; }
  else if (t.dataset.s != null) { S.s = t.dataset.s; }
  else if (t.dataset.p != null) { S.p = t.dataset.p; }
  else if (t.dataset.go) {
    const g = t.dataset.go;
    return setView('stream', 'all', g, {range: '14'});
  }
  shown = PAGE; update(true);
});
document.addEventListener('toggle', async (ev) => {
  const d = ev.target;
  if (!(d instanceof HTMLDetailsElement) || !d.open || !d.dataset.abs || d.dataset.done) return;
  d.dataset.done = '1';
  const it = items.find((i) => String(i.id) === d.dataset.abs);
  const p = d.querySelector('p');
  try { p.textContent = (it && await abstractOf(it)) || '(no abstract available)'; } catch (e) { p.textContent = '(could not load abstract)'; }
}, true);
$('#onlysaved').addEventListener('change', (e) => { S.sv = e.target.checked ? '1' : '0'; shown = PAGE; update(true); });
$('#hideread').addEventListener('change', (e) => { S.hr = e.target.checked ? '1' : '0'; shown = PAGE; update(true); });
$('#range').addEventListener('change', (e) => { S.range = e.target.value; shown = PAGE; update(true); });
$('#sort').addEventListener('change', (e) => { S.sort = e.target.value; update(true); });
$('#rel').addEventListener('input', (e) => { S.rel = e.target.value; $('#relval').textContent = S.rel; shown = PAGE; renderList(); saveState(); });
let qt = null;
$('#q').addEventListener('input', (e) => { S.q = e.target.value; clearTimeout(qt); qt = setTimeout(() => { shown = PAGE; update(true); }, 250); });
window.addEventListener('hashchange', () => { const h = parseHash(); if (Object.keys(h).length) { S = {...S, ...h}; shown = PAGE; update(false); } });
window.addEventListener('error', (e) => showErr(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => showErr(e.reason));

// ---------------------------------------------------------------- By person: people you follow
const PERSON_STREAMS = ['literature', 'people', 'reporter'];
const hasIds = (p) => !!(p.openalex_id && (!Array.isArray(p.openalex_id) || p.openalex_id.length));
async function loadWatch() { watchList = (await getJSON('api/watch')).people || []; }
const shortName = (n) => { const [l, f] = String(n).split(', '); return f ? `${l}, ${f[0]}.` : l; };
function personCounts() {
  const cutoff = Date.now() - (+S.range) * 864e5, c = {};
  watchList.forEach((w) => { c[w.name] = 0; });
  items.forEach((it) => { watchList.forEach((w) => { if (passes(it, cutoff, status.today, w.name)) c[w.name] += 1; }); });
  return c;
}
function renderPersonTabs() {
  const c = personCounts(), total = Object.values(c).reduce((a, b) => a + b, 0);
  const people = [...watchList].sort((a, b) => c[b.name] - c[a.name] || a.name.localeCompare(b.name));
  const chip = (id, lab, n) => `<button role="tab" aria-selected="${id === S.p}" class="${id === S.p ? 'on' : ''}${n === 0 ? ' zero' : ''}" data-p="${esc(id)}">${esc(lab)}${n != null ? ' · ' + n : ''}</button>`;
  $('#primary').innerHTML = chip('all', 'All followed', total) + people.map((w) => chip(w.name, shortName(w.name), c[w.name])).join('');
}
function renderPerson() {
  const el = $('#person'); if (!el) return;
  el.hidden = S.view !== 'person';
  if (el.hidden) return;
  if (!api) { el.innerHTML = '<div class="note">Following people needs the server API, which is not available in this static view.</div>'; return; }
  if (S.p !== 'all' && !watchList.some((w) => w.name === S.p)) S.p = 'all';
  const sel = watchList.find((w) => w.name === S.p);
  const missing = watchList.filter((p) => !hasIds(p)).length;
  const header = sel ? `<article class="card pcard"><div class="t">👤 ${esc(sel.name)}</div>
      <div class="kv"><span>NIH awards profile <b>${sel.profile_id ? '✓ linked' : 'none'}</b></span>
      <span>Publications <b>${hasIds(sel) ? '✓ ' + (Array.isArray(sel.openalex_id) ? sel.openalex_id.length + ' records' : 'linked') : 'not linked'}</b></span></div>
      <div class="acts"><button class="pill danger" data-unfollow="${esc(sel.name)}">Unfollow ${esc(sel.name)}</button></div></article>`
    : `<div class="note">${watchList.length} people followed. Their new NIH awards appear under Funded Projects and their new papers under People; tap a name to see just them.${missing ? ` <span class="warn">${missing} have no publications id, so no papers are tracked for them.</span>` : ''}</div>`;
  el.innerHTML = `${header}
    <details id="followbox" class="followbox" ${followOpen ? 'open' : ''}><summary>＋ Follow someone new · or manage the list</summary>
      <div class="wbody"><form id="wform" class="wform"><input id="wq" type="search" placeholder="Last, First  or  First Last" autocomplete="off"><button type="submit" class="pill on">Search</button></form>
      <div id="wresults"></div>
      <div class="wh">Followed (${watchList.length})</div>
      <ul class="wlist">${watchList.map((p) => `<li><span>${esc(p.name)}${hasIds(p) ? '' : ' <span class="badge b-amber">no pubs id</span>'}${p.profile_id ? '' : ' <span class="badge b-gray">no NIH profile</span>'}</span>
        <button class="pill danger" data-unfollow="${esc(p.name)}">Unfollow</button></li>`).join('')}</ul></div></details>`;
  const box = $('#followbox');
  box.addEventListener('toggle', () => { followOpen = box.open; });
  $('#wform').addEventListener('submit', (e) => { e.preventDefault(); searchWatch($('#wq').value); });
}
async function searchWatch(q) {
  const box = $('#wresults'); box.innerHTML = '<div class="note">Searching NIH RePORTER and OpenAlex…</div>';
  try {
    const r = await getJSON('api/watch/search?q=' + encodeURIComponent(q));
    const rep = r.reporter || [], oa = r.openalex || [];
    const top = oa[0] && oa[0].institutions[0];
    box.innerHTML = `<div class="wres" data-name="${esc(r.name)}"><b>${esc(r.name)}</b>
      <div class="wh">NIH awards (RePORTER profile) <span class="note">pick one, or none</span></div>
      ${rep.map((c, i) => `<label class="opt"><input type="radio" name="wrp" value="${esc(c.profile_id)}" ${i === 0 && rep.length === 1 ? 'checked' : ''}> ${esc(c.name)} · ${c.awards} award${c.awards === 1 ? '' : 's'} · ${esc(c.institutions.join('; '))}</label>`).join('')
        || '<div class="note">No NIH PI profile found (they may not hold NIH awards).</div>'}
      <label class="opt"><input type="radio" name="wrp" value="" ${rep.length === 1 ? '' : 'checked'}> none</label>
      <div class="wh">Publications (OpenAlex) <span class="note">tick every record that is this person</span></div>
      ${oa.map((c) => `<label class="opt"><input type="checkbox" name="woa" value="${esc(c.id)}" ${c.institutions[0] && c.institutions[0] === top && c.works > 0 ? 'checked' : ''}> ${esc(c.name)} · ${c.works} works · ${esc(c.institutions.join('; ') || 'no institution')}</label>`).join('')
        || '<div class="note">No OpenAlex author found.</div>'}
      ${(r.errors || []).map((e) => `<div class="note">⚠ ${esc(e)}</div>`).join('')}
      <button class="pill on" data-wadd="1">＋ Follow</button></div>`;
  } catch (e) { box.innerHTML = `<div class="note">${esc(e.message)}</div>`; }
}
async function followPerson(entry) {
  const r = await apiCall('POST', 'api/watch/add', entry);
  watchList = r.people || [];
  if (r.added && S.view === 'person' && watchList.length) S.p = watchList[watchList.length - 1].name;
  followOpen = false; await update(true);
}
async function unfollowPerson(name) {
  const r = await apiCall('POST', 'api/watch/remove', {name});
  watchList = r.people || [];
  if (S.p === name) S.p = 'all';
  await update(true);
}
async function watchPi(btn) {                              // "＋ Follow" on an award card
  btn.disabled = true;
  try { await followPerson({name: btn.dataset.name, profile_id: +btn.dataset.watchPi}); btn.textContent = '✓ Following'; }
  catch (e) { btn.disabled = false; showErr(e); }
}
document.addEventListener('click', async (ev) => {
  const add = ev.target.closest('[data-wadd]');
  if (add) {
    const box = add.closest('.wres'), rp = box.querySelector('input[name=wrp]:checked'), oa = [...box.querySelectorAll('input[name=woa]:checked')].map((i) => i.value);
    const entry = {name: box.dataset.name};
    if (rp && rp.value) entry.profile_id = +rp.value;
    if (oa.length) entry.openalex_id = oa;
    add.disabled = true;
    try { await followPerson(entry); } catch (e) { add.disabled = false; showErr(e); }
    return;
  }
  const un = ev.target.closest('[data-unfollow]');
  if (un && confirm('Unfollow ' + un.dataset.unfollow + '?')) { try { await unfollowPerson(un.dataset.unfollow); } catch (e) { showErr(e); } }
});

// ---------------------------------------------------------------- briefing pop-up (daily + weekly, written by Claude)
const localAcks = () => { try { return new Set(JSON.parse(store.get('lh:acks') || '[]')); } catch (e) { return new Set(); } };
const isAcked = (key) => acked.has(key) || localAcks().has(key);
async function loadBriefing() { try { briefing = await getJSON('data/briefing.json'); } catch (e) { briefing = {daily: null, weekly: null}; } }
function briefingValid(b) {                                   // daily: its own day; weekly: until three days later
  if (!b || !b.date || !status) return false;
  const age = Math.round((new Date(status.today + 'T00:00:00') - new Date(b.date + 'T00:00:00')) / 864e5);
  return b.period === 'daily' ? age === 0 : age >= 0 && age <= 3;
}
function citeText(text, order, byId) {
  return esc(text).replace(/\[i(\d+)\]/g, (m, id) => {
    const it = byId.get(+id); if (!it) return '';                                 // cited item not loaded: drop the link quietly
    if (!order.has(+id)) order.set(+id, order.size + 1);
    return `<a class="cite" href="${esc(safeUrl(it.url))}" title="${esc(it.title)}" data-cite="${id}">[${order.get(+id)}]</a>`;
  }).replace(/\s+([.,;:])/g, '$1');
}
function renderBriefing() {
  const el = $('#briefing'), link = $('#brief-link');
  const all = [briefing.weekly, briefing.daily].filter(Boolean);
  if (!el) return;
  if (link) link.hidden = !all.length;
  const show = all.filter((b) => briefingValid(b) && (showAllBriefings || !isAcked(b.key)));
  if (!show.length) { el.hidden = true; el.innerHTML = ''; return; }
  const byId = new Map(items.map((i) => [i.id, i]));
  el.hidden = false;
  el.innerHTML = show.map((b) => {
    const order = new Map();
    const rich = (t) => citeText(t, order, byId).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');   // text is escaped first, so this stays safe
    const secs = (b.sections && b.sections.length) ? b.sections : [{heading: '', paragraphs: b.paragraphs || [], bullets: []}];   // older briefings: flat paragraphs
    const paras = secs.map((sec) => (sec.heading ? `<h3 class="bs">${esc(sec.heading)}</h3>` : '')
      + (sec.paragraphs || []).map((p) => `<p>${rich(p)}</p>`).join('')
      + ((sec.bullets || []).length ? `<ul class="bl">${sec.bullets.map((x) => `<li>${rich(x)}</li>`).join('')}</ul>` : '')).join('');
    const refs = [...order.entries()].map(([id, n]) => { const it = byId.get(id); return `<li value="${n}"><a href="${esc(safeUrl(it.url))}">${esc(it.title)}</a> <span class="meta">${esc(SHORT[it.stream] || it.stream)}</span></li>`; }).join('');
    const c = b.counts || {};
    return `<article class="brief ${b.period}" data-key="${esc(b.key)}">
      <div class="bh">${b.period === 'weekly' ? '📅' : '☀️'} ${esc(b.title)}</div>
      ${b.headline ? `<div class="bt">${esc(b.headline)}</div>` : ''}${paras}
      ${refs ? `<details class="bsrc"><summary>Items mentioned (${order.size})</summary><ol>${refs}</ol></details>` : ''}
      <div class="bf"><span class="note">Written by Claude${b.model ? ' (' + esc(b.model) + ')' : ''} from ${c.items_given || '?'} of ${c.items_total || '?'} new items · ${esc(ago(b.generated))}</span>
      <button class="pill on" data-ack="${esc(b.key)}">${showAllBriefings && isAcked(b.key) ? 'Close' : 'Got it'}</button></div></article>`;
  }).join('');
}
async function ackBriefing(key) {
  acked.add(key);
  const l = [...localAcks(), key].slice(-60); store.set('lh:acks', JSON.stringify(l));       // also remembered per device if the API is down
  if (api) apiCall('POST', 'api/ack', {key}).catch(() => {});
  renderBriefing();
}
document.addEventListener('click', (ev) => {
  const a = ev.target.closest('[data-ack]');
  if (a) { ackBriefing(a.dataset.ack); return; }
  if (ev.target.closest('#brief-link')) { ev.preventDefault(); showAllBriefings = !showAllBriefings; renderBriefing(); }
});

// ---------------------------------------------------------------- audio briefing (two-host podcast, voiced locally on the Pi; made on demand)
async function loadPodcast() { try { podcast = await getJSON('data/podcast.json'); } catch (e) { podcast = {episodes: []}; } }
async function loadPodApi() {
  if (!api || apiV < 3) { podApi = null; return; }
  try { podApi = await getJSON('api/podcast'); } catch (e) { podApi = null; }
}
const fmtDur = (s) => (s ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : '');
function podLastText(l) {
  if (!l) return '';
  const notes = (l.notes || []).filter((n) => /^podcast|deep dive/.test(n)).map((n) => n.replace(/^podcast:\s*/, ''));
  return l.ok ? notes.join(' · ') : `Last attempt failed: ${l.error || 'see the status panel'}`;
}
let podTopic = 'auto', podCustom = '';
function topicPicker(a) {
  const byArea = {};
  (a.topics || []).forEach((t) => { (byArea[t.area] = byArea[t.area] || []).push(t); });
  const opt = (v, label) => `<option value="${esc(v)}"${podTopic === v ? ' selected' : ''}>${esc(label)}</option>`;
  const groups = Object.keys(byArea).sort().map((ar) => `<optgroup label="${esc(ar)}">${byArea[ar].map((t) => opt(t.id, t.title + (t.days_ago != null ? ` (covered ${t.days_ago} d ago)` : ''))).join('')}</optgroup>`).join('');
  return `<div class="pod-topic"><label class="note">Methods segment <select id="pod-topic">${opt('auto', 'Automatic (linked to today\'s papers)')}${opt('none', 'None (skip it)')}${opt('custom', 'Custom topic…')}${groups}</select></label>
    <input id="pod-custom" type="text" maxlength="120" placeholder="e.g. propensity score matching" value="${esc(podCustom)}" ${podTopic === 'custom' ? '' : 'hidden'}></div>`;
}
function renderPodControls() {
  const el = $('#pod-ctl'); if (!el) return;
  const a = podApi;
  if (!a || !a.enabled) { el.innerHTML = ''; return; }
  const cost = '$' + (a.est_cost || 0.3).toFixed(2), left = a.cap != null ? Math.max(0, a.cap - a.spend_today) : null;
  let btn, msg = '';
  if (a.running) {
    btn = '<button class="pill" disabled>⏳ Generating…</button>';
    msg = 'Writing the script and recording it. This takes about 5 to 6 minutes; you can leave this page open or come back.';
  } else {
    btn = `<button class="pill on" data-podgen="${a.episode_today ? 'redo' : 'new'}">${a.episode_today ? '↻ Regenerate' : '🎙 Generate'} today's episode (~${cost})</button>`
      + (a.methods && (a.topics || []).length ? topicPicker(a) : '');
    msg = (left != null ? `Podcast budget today: $${a.spend_today.toFixed(2)} of $${a.cap.toFixed(2)} used (separate from the hub budget). ` : '') + podLastText(a.last);
  }
  el.innerHTML = `<div class="pod-gen">${btn}</div>${msg ? `<div class="note">${esc(msg)}</div>` : ''}`;
}
function renderPodcast(force) {
  const el = $('#podcast'), link = $('#pod-link');
  if (!el) return;
  const eps = podcast.episodes || [], canGen = !!(podApi && podApi.enabled);
  if (link) link.hidden = !(eps.length || canGen);
  if (!eps.length && !canGen) { el.hidden = true; el.innerHTML = ''; podSig = ''; return; }
  el.hidden = false;
  if (!$('#pod-ctl')) { el.innerHTML = '<div id="pod-ctl"></div><div id="pod-ep"></div>'; podSig = ''; }
  renderPodControls();
  const box = $('#pod-ep');
  if (!eps.length) { box.innerHTML = '<div class="note">No episodes yet.</div>'; podSig = ''; return; }
  const sig = eps.map((e) => e.key).join(',') + '|' + podIdx;
  if (!force && sig === podSig) return;                          // never rebuild while the audio element is playing
  podSig = sig;
  const ep = eps[Math.min(podIdx, eps.length - 1)];
  const byId = new Map(items.map((i) => [i.id, i])), order = new Map();
  const names = ep.hosts || {};
  let seg = 'news';
  const lines = (ep.turns || []).map((t) => {
    const head = (t.segment === 'deep_dive' && seg !== 'deep_dive') ? `<h3 class="bs">Deep dive${ep.dive ? ': ' + esc(ep.dive.title) : ''}</h3>`
      : (t.segment === 'methods' && seg !== 'methods') ? `<h3 class="bs">Methods${ep.methods ? ': ' + esc(ep.methods.title) : ''}</h3>` : '';
    seg = t.segment || 'news';
    return head + `<p><span class="spk">${esc(names[t.speaker] || t.speaker)}</span> ${citeText(t.text, order, byId)}</p>`;
  }).join('');
  const refs = [...order.entries()].map(([id, n]) => { const it = byId.get(id); return `<li value="${n}"><a href="${esc(safeUrl(it.url))}">${esc(it.title)}</a> <span class="meta">${esc(SHORT[it.stream] || it.stream)}</span></li>`; }).join('');
  const pick = eps.length > 1 ? `<select id="pod-pick" aria-label="Episode">${eps.map((e, i) => `<option value="${i}"${i === podIdx ? ' selected' : ''}>${esc(e.title)}</option>`).join('')}</select>` : '';
  box.innerHTML = `<article class="pod">
    <div class="bh">🎧 ${esc(ep.title)}${ep.seconds ? ' · ' + fmtDur(ep.seconds) : ''}</div>
    ${ep.headline ? `<div class="bt">${esc(ep.headline)}</div>` : ''}
    ${ep.dive ? `<div class="note">🔬 Deep dive: <a href="${esc(safeUrl(ep.dive.url))}">${esc(ep.dive.title)}</a> · full text via ${esc(ep.dive.source)}</div>` : ''}
    <audio id="pod-audio" controls preload="none" src="${esc(ep.audio)}"></audio>
    <div class="pod-row"><span class="note">Speed</span>${[1, 1.25, 1.5, 1.75].map((s) => `<button class="pill${s === podSpeed ? ' on' : ''}" data-speed="${s}">${s}×</button>`).join('')}${pick}</div>
    ${ep.methods ? `<div class="note">📐 Methods: ${esc(ep.methods.title)}${ep.methods.how === 'linked' ? ' · linked to today\'s papers' : ''}</div>` : ''}
    <details class="bsrc"><summary>Transcript and links${order.size ? ` (${order.size} items)` : ''}</summary><div class="transcript">${lines}</div>${refs ? `<ol>${refs}</ol>` : ''}</details>
    ${methodsNotes(ep)}
    <div class="note">Written by Claude${ep.model ? ' (' + esc(ep.model) + ')' : ''}, read by synthetic voices · ${esc(ago(ep.generated))}</div></article>`;
  const a = $('#pod-audio');
  if (a) {
    a.playbackRate = podSpeed;
    a.addEventListener('play', () => { a.playbackRate = podSpeed; });
    if ('mediaSession' in navigator && window.MediaMetadata) {
      navigator.mediaSession.metadata = new MediaMetadata({title: ep.headline || ep.title, artist: 'Research hub', album: ep.title});
    }
  }
}
function methodsNotes(ep) {
  const m = ep.methods, n = m && m.notes;
  if (!n || !(((n.summary || []).length) || ((n.formulas || []).length) || ((n.pitfalls || []).length))) return '';
  const li = (a) => (a || []).map((x) => `<li>${esc(x)}</li>`).join('');
  return `<details class="bsrc mnotes"><summary>Methods notes: ${esc(m.title)}</summary>
    ${(n.summary || []).length ? `<h4>Key points</h4><ul>${li(n.summary)}</ul>` : ''}
    ${(n.formulas || []).length ? `<h4>Formulas</h4><ul class="formulas">${n.formulas.map((f) => `<li><b>${esc(f.name)}</b> <code>${esc(f.text)}</code></li>`).join('')}</ul>` : ''}
    ${(n.pitfalls || []).length ? `<h4>Common pitfalls</h4><ul>${li(n.pitfalls)}</ul>` : ''}
    ${m.verified_example ? '<div class="note">The worked-example numbers in this segment were computed by code.</div>' : ''}</details>`;
}
let podTimer = null;
function podPoll() {
  clearTimeout(podTimer);
  podTimer = setTimeout(async () => {
    const was = podApi && podApi.running;
    await loadPodApi();
    if (podApi && podApi.running) { renderPodControls(); podPoll(); return; }
    if (was) { await loadPodcast(); renderPodcast(true); try { status = await getJSON('data/status.json'); renderStatus(); } catch (e) { /* ignore */ } }
    renderPodControls();
  }, 6000);
}
async function generatePodcast(redo) {
  const a = podApi; if (!a) return;
  const left = a.cap != null ? ` Podcast budget used today: $${a.spend_today.toFixed(2)} of $${a.cap.toFixed(2)}.` : '';
  const sel = $('#pod-topic'), custom = $('#pod-custom');
  if (sel) podTopic = sel.value;
  if (custom) podCustom = custom.value.trim();
  if (podTopic === 'custom' && !podCustom) { showErr(new Error('Type a topic for the methods segment, or choose Automatic.')); return; }
  const topic = podTopic === 'custom' ? 'custom:' + podCustom : podTopic;
  if (!confirm(`${redo ? 'Replace today\'s episode with a new one?' : 'Generate today\'s episode?'} It costs about $${(a.est_cost || 0.3).toFixed(2)} and takes 5 to 6 minutes.${left}`)) return;
  try {
    await apiCall('POST', 'api/podcast/generate', {redo: !!redo, topic});
    podApi = {...a, running: true, started: new Date().toISOString()};
    renderPodControls(); podPoll();
  } catch (e) { showErr(e); await loadPodApi(); renderPodControls(); }
}
document.addEventListener('click', (ev) => {
  const sp = ev.target.closest('[data-speed]');
  if (sp) { podSpeed = +sp.dataset.speed; const a = $('#pod-audio'); if (a) a.playbackRate = podSpeed;
    document.querySelectorAll('[data-speed]').forEach((b) => b.classList.toggle('on', +b.dataset.speed === podSpeed)); return; }
  const g = ev.target.closest('[data-podgen]');
  if (g) { generatePodcast(g.dataset.podgen === 'redo'); return; }
  if (ev.target.closest('#pod-link')) { ev.preventDefault(); const el = $('#podcast'); if (el && !el.hidden) el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
});
document.addEventListener('change', (ev) => {
  if (ev.target && ev.target.id === 'pod-topic') { podTopic = ev.target.value; const c = $('#pod-custom'); if (c) { c.hidden = podTopic !== 'custom'; if (!c.hidden) c.focus(); } }
  if (ev.target && ev.target.id === 'pod-pick') { podIdx = +ev.target.value; renderPodcast(true); }
});

// ---------------------------------------------------------------- tuning: proposals from your ratings (you approve each)
async function loadTuning() {
  if (!api || apiV < 2) return;                       // older server: the tuning endpoints don't exist yet
  try { tuningState = await getJSON('api/tuning'); } catch (e) { tuningState = null; }
  renderTuningLink();
}
function renderTuningLink() {
  const a = $('#tune-link'); if (!a) return;
  a.hidden = !api || !tuningState;
  const n = tuningState ? tuningState.pending.length : 0;
  a.textContent = '🎛 Tuning' + (n ? ` (${n})` : '');
  a.style.fontWeight = n ? '700' : '400';
}
async function refreshItems() {                       // an accepted preference changes each item's adjustment: reload the data
  try {
    const recent = await getJSON('data/items.json'); items = recent.items;
    if (fullLoaded) { items = (await getJSON('data/items-90d.json')).items; }
  } catch (e) { showErr(e); }
}
const fmtW = (w) => (w > 0 ? '+' : '') + w;
function evidenceHtml(ev) {
  const byId = new Map(items.map((i) => [i.id, i]));
  return (ev || []).length ? `<div class="pev"><span class="note">Based on:</span> ${(ev || []).map((e) => { const it = byId.get(e.id);
    const t = `${e.sign > 0 ? '👍' : e.sign < 0 ? '👎' : ''} ${esc(e.title.length > 70 ? e.title.slice(0, 67) + '…' : e.title)}`;
    return it ? `<a href="${esc(safeUrl(it.url))}">${t}</a>` : `<span>${t}</span>`; }).join(' · ')}</div>` : '';
}
function proposalHtml(p, history) {
  const pl = p.payload || {}, bt = p.backtest || {};
  let head = '', body = '';
  if (p.kind === 'preferences') {
    head = '✎ Update your learned preferences paragraph';
    body = `<div class="pdiff"><div><div class="wh">Current</div><div class="ptext">${esc(pl.old || '(none yet)')}</div></div><div><div class="wh">Proposed</div><div class="ptext new">${esc(pl.new)}</div></div></div>`;
  } else if (p.kind === 'rule') {
    head = `${pl.weight < 0 ? '↓ Deprioritize' : '↑ Prefer'} ${esc(pl.kind)} “${esc(pl.value)}” <span class="badge ${pl.weight < 0 ? 'b-amber' : 'b-green'}">${fmtW(pl.weight)}</span>`;
    body = `<div class="meta">You rated ${pl.likes + pl.dislikes} items with this: ${pl.likes} 👍 / ${pl.dislikes} 👎.</div>
      ${bt.matches !== undefined ? `<div class="meta">Backtest: it would have touched <b>${bt.matches}</b> item${bt.matches === 1 ? '' : 's'} in the last ${bt.days} days (${bt.liked} you liked, ${bt.disliked} you disliked, ${bt.unrated} unrated). Ranking only; nothing is hidden.</div>` : ''}`;
  } else if (p.kind === 'topic') {
    head = `📐 Add a methods topic: ${esc(pl.title)}`;
    body = `<div class="meta">${esc(pl.area || '')} · linked to papers by these keywords: ${(pl.keywords || []).map(esc).join(', ')}</div><div class="ptext">${esc(pl.scope || '')}</div>`;
  } else {
    head = `👤 Follow ${esc(pl.name)}?`;
    body = `<div class="meta">On ${pl.likes} paper${pl.likes === 1 ? '' : 's'} you liked.</div>`;
  }
  const btns = history ? (p.status === 'accepted' && p.kind !== 'follow' ? `<button class="pill" data-tdecide="undo" data-pid="${p.id}">Undo</button>` : '')
    : p.kind === 'follow' ? `<button class="pill on" data-tfollow="${esc(pl.name)}">Search &amp; follow</button><button class="pill" data-tdecide="reject" data-pid="${p.id}">Dismiss</button>`
    : `<button class="pill on" data-tdecide="accept" data-pid="${p.id}">Accept</button><button class="pill" data-tdecide="reject" data-pid="${p.id}">Reject</button>`;
  return `<article class="prop ${history ? 'hist' : ''}"><div class="t">${head}${history ? ` <span class="badge b-gray">${esc(p.status)}</span>` : ''}</div>${p.why ? `<div class="why">${esc(p.why)}</div>` : ''}${body}${evidenceHtml(p.evidence_items)}<div class="acts">${btns}</div></article>`;
}
function renderTuning() {
  const el = $('#tuning'); if (!el) return;
  if (!tuningOpen || !tuningState) { el.hidden = true; el.innerHTML = ''; return; }
  const t = tuningState;
  el.hidden = false;
  el.innerHTML = `<div class="tuneh">🎛 Tuning <button class="x" data-tclose="1" aria-label="Close">✕</button></div>
    <div class="note">${t.ratings} ratings so far · ${t.new_ratings} of ${t.need} new ratings toward the next tune-up. Nothing changes until you accept; every accepted change can be undone, and it only re-ranks (nothing is hidden).</div>
    ${t.notes ? `<div class="tnotes"><b>What the last tune-up noticed:</b> ${esc(t.notes)}</div>` : ''}
    ${t.pending.length ? t.pending.map((p) => proposalHtml(p, false)).join('') : '<div class="empty">No proposals waiting.</div>'}
    ${t.history.length ? `<details class="bsrc"><summary>Recent decisions (${t.history.length})</summary>${t.history.map((p) => proposalHtml(p, true)).join('')}</details>` : ''}`;
}
async function decideProposal(pid, action) {
  try {
    await apiCall('POST', 'api/tuning/decide', {id: +pid, action});
    await loadTuning();
    if (action !== 'reject') await refreshItems();
    renderTuning(); renderList();
  } catch (e) { showErr(e); }
}
function followFromSuggestion(name) {               // PubMed-style 'Smith J' -> 'Smith, J' for the search box
  const m = /^(\S+) ([A-Z]{1,3})$/.exec(name);
  const q = m ? `${m[1]}, ${m[2]}` : name;
  tuningOpen = false; renderTuning(); followOpen = true; S.view = 'person'; S.p = 'all';
  update(true).then(() => { const box = $('#wq'); if (box) { box.value = q; searchWatch(q); window.scrollTo({top: $('.controls').offsetTop - 4, behavior: 'smooth'}); } });
}
document.addEventListener('click', (ev) => {
  const d = ev.target.closest('[data-tdecide]');
  if (d) { decideProposal(d.dataset.pid, d.dataset.tdecide); return; }
  const f = ev.target.closest('[data-tfollow]');
  if (f) { followFromSuggestion(f.dataset.tfollow); return; }
  if (ev.target.closest('[data-tclose]')) { tuningOpen = false; renderTuning(); return; }
  if (ev.target.closest('#tune-link')) { ev.preventDefault(); tuningOpen = !tuningOpen; renderTuning(); if (tuningOpen) window.scrollTo({top: 0, behavior: 'smooth'}); }
});

// ---------------------------------------------------------------- stale-page guard
// The script and index.html ship together. If the browser kept an old index.html, elements this script needs are missing.
const REQUIRED_IDS = ['err', 'status', 'whatsnew', 'briefing', 'brief-link', 'podcast', 'pod-link', 'tuning', 'tune-link', 'person', 'v-domain', 'v-stream', 'v-person', 'primary', 'secondary',
  'range', 'sort', 'rel', 'relval', 'q', 'count', 'toggles', 'onlysaved', 'hideread', 'list', 'more', 'foot'];
function pageIsStale() { return REQUIRED_IDS.filter((id) => !document.getElementById(id)); }
function healStalePage(missing) {
  let tried = false;
  try { tried = sessionStorage.getItem('lh:healed') === '1'; sessionStorage.setItem('lh:healed', '1'); } catch (e) { /* ignore */ }
  const e = document.getElementById('err') || document.body.insertBefore(document.createElement('div'), document.body.firstChild);
  e.hidden = false; e.className = 'err';
  e.innerHTML = `Your browser has an out-of-date copy of this page (missing: ${missing.join(', ')}). ${tried ? 'Reload did not fix it: <a href="?r=' + Date.now() + '">tap here</a> to load the current version.' : 'Reloading…'}`;
  if (!tried) location.replace(location.pathname + '?r=' + Date.now() + location.hash);   // new URL = never served from the cache
}

// ---------------------------------------------------------------- boot
(async function init() {
  const stale = pageIsStale();
  if (stale.length) { healStalePage(stale); return; }
  try { sessionStorage.removeItem('lh:healed'); } catch (e) { /* ignore */ }
  loadState();
  try {
    prev = store.sget('lh:prev');
    if (prev === null) { prev = store.get('lh:last') || 'none'; store.sset('lh:prev', prev); }
    if (prev === 'none') prev = null;
    store.set('lh:last', new Date().toISOString());
  } catch (e) { prev = null; }
  try {
    const [st, it] = await Promise.all([getJSON('data/status.json'), getJSON('data/items.json')]);
    status = st; items = it.items;
    await initApi();
    if (api) { try { await loadWatch(); } catch (e) { watchList = []; } }
    await loadBriefing();
    await loadPodcast();
    await loadPodApi();
    await loadTuning();
    renderBriefing(); renderPodcast();
    if (podApi && podApi.running) podPoll(); renderStatus(); renderWhatsNew();
    await update(false);
  } catch (e) {
    showErr(e); $('#list').innerHTML = '<div class="empty">No data yet. Run a stream (or <code>lit build-site</code>) to generate it.</div>';
  }
})();
})();
