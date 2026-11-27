#!/usr/bin/env python3
"""Competitor tracker for code-made simulation, quiz, chess and map Shorts.

Standard library only. One cheap run per day (about 2 requests per channel):

    python3 research/track.py                    # snapshot for today (UTC) + summary
    python3 research/track.py --date 2026-11-17  # label the snapshot with another date
    python3 research/track.py --add HANDLE       # append a channel to watchlist.json
    python3 research/track.py --summary-only     # rebuild the summary, no network

Per channel: the About page (subscribers, total views, video count, exact) and
the RSS feed (15 latest videos, exact views and publish times). When the RSS
feed fails, the public uploads playlist page is used instead; its view counts
and ages are rounded by YouTube, so those values are marked approximate ("~").

Outputs (next to this file): watchlist.json (channel ids cached),
data/YYYY-MM-DD.json (snapshot), latest_summary.md (at most 70 lines).
"""
import argparse
import datetime as dt
import http.client
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PACE_S = 0.5            # pause between requests
TIMEOUT_S = 15          # socket timeout per request
SLOW_LIMIT_S = 25       # wall-clock limit for one response
BUDGET_S = 215          # stop fetching after this long (whole run stays under 4 minutes)
MAX_BYTES = 12_000_000  # response size cap (decoded)
PAGE_VIDEOS = 30        # videos kept per channel when the playlist page is the source
RSS_FAILS_TO_DISABLE = 3
SUMMARY_MAX_LINES = 70
TOP_PER_CHANNEL = 5      # rows per channel in the top-20 table, so one channel cannot fill it
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US",
           "Accept-Encoding": "gzip", "Cookie": "SOCS=CAI"}
ABOUT_BY_ID = "https://www.youtube.com/channel/{}/about"
ABOUT_BY_HANDLE = "https://www.youtube.com/@{}/about"
RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
UPLOADS_URL = "https://www.youtube.com/playlist?list={}"
NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015",
      "media": "http://search.yahoo.com/mrss/"}
STOP = set("""a an and are as at be but by can do for from get has have how i if in into is it its
me my no not of on or our out so than that the their them then there these they this to up was
we what when which who why will with you your he she his her him us shorts short""".split())
GENERIC_TAGS = {"#shorts", "#short", "#ytshorts", "#youtubeshorts", "#shortsfeed"}


class FetchError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ParseError(Exception):
    pass


# ----------------------------------------------------------------------------- network

class Net:
    """Paced GET with timeouts, one retry on transient errors and a global deadline."""

    def __init__(self, deadline):
        self.deadline = deadline
        self.last = 0.0
        self.requests = 0

    def get(self, url, retries=1):
        attempt = 0
        while True:
            remaining = self.deadline - time.time()
            if remaining < 3:
                raise FetchError("time budget exhausted")
            wait = self.last + PACE_S - time.time()
            if wait > 0:
                time.sleep(wait)
            self.requests += 1
            try:
                return self._once(url, min(TIMEOUT_S, remaining))
            except FetchError as e:
                transient = e.code in (None, 429, 500, 502, 503, 504)
                if not transient or attempt >= retries:
                    raise
                attempt += 1
                time.sleep(4 if e.code == 429 else 1.5)
            finally:
                self.last = time.time()

    @staticmethod
    def _once(url, timeout):
        req = urllib.request.Request(url, headers=HEADERS)
        t_end = time.time() + SLOW_LIMIT_S
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                buf = bytearray()
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > MAX_BYTES:
                        raise FetchError("response too large", 0)
                    if time.time() > t_end:
                        raise FetchError("response too slow")
                if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    buf = d.decompress(bytes(buf), MAX_BYTES)
                    if not d.eof and not d.unconsumed_tail:
                        raise FetchError("truncated response")   # transfer cut off: retried once
                return buf.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise FetchError("HTTP %d" % e.code, e.code) from None
        except FetchError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, OSError, zlib.error, ValueError) as e:
            raise FetchError("%s: %s" % (type(e).__name__, str(e)[:120])) from None


# ----------------------------------------------------------------------------- parsing

def initial_data(html):
    m = re.search(r"var ytInitialData\s*=\s*(\{.*?\})\s*;\s*</script>", html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def walk(obj, key):
    """Yield every value stored under `key`, in document order."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, key)


_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([KMB])?(?![A-Za-z])", re.I)


def parse_count(text):
    """'8,582,950 views' -> (8582950, 0); '4.28K subscribers' -> (4280, 10); '19K' -> (19000, 1000).

    The second value is the rounding step (0 = exact). YouTube appears to truncate
    abbreviated counts, so the true value is assumed to be in [value, value + step).
    """
    if not text:
        return None, 0
    t = str(text).replace("\xa0", " ").strip()
    if re.match(r"(?i)no\b", t):
        return 0, 0
    m = _NUM.search(t)
    if not m:
        return None, 0
    num, suf = m.group(1).replace(",", ""), (m.group(2) or "").upper()
    if not suf:
        return int(float(num)), 0
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suf]
    decimals = len(num.split(".")[1]) if "." in num else 0
    return int(round(float(num) * mult)), max(1, mult // 10 ** decimals)


_AGE = re.compile(r"(\d+)\s*(second|sec|minute|min|hour|hr|day|week|wk|month|mo|year|yr|s|m|h|d|w|y)s?\b", re.I)
_AGE_HOURS = {"second": 1 / 3600, "sec": 1 / 3600, "s": 1 / 3600, "minute": 1 / 60, "min": 1 / 60,
              "m": 1 / 60, "hour": 1, "hr": 1, "h": 1, "day": 24, "d": 24, "week": 168, "wk": 168,
              "w": 168, "month": 730, "mo": 730, "year": 8766, "yr": 8766, "y": 8766}


def parse_age_hours(text):
    """'6h ago' -> 6.5, '2d ago' -> 60.0 (middle of the range the label covers). Heuristic."""
    m = _AGE.search((text or "").replace("\xa0", " "))
    if not m:
        return None
    return (int(m.group(1)) + 0.5) * _AGE_HOURS[m.group(2).lower()]


def parse_duration(text):
    if not text or not re.fullmatch(r"\d+(:\d\d){1,2}", text.strip()):
        return None
    sec = 0
    for part in text.strip().split(":"):
        sec = sec * 60 + int(part)
    return sec


def iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts_of(text):
    if not text:
        return None
    try:
        d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def parse_about(html):
    """Channel stats from an About page. Total views are exact; subscribers are rounded by YouTube."""
    data = initial_data(html)
    about = next(walk(data, "aboutChannelViewModel"), None) if data else None
    meta = next(walk(data, "channelMetadataRenderer"), None) if data else None
    if not isinstance(about, dict):
        about = {}
        for key in ("subscriberCountText", "viewCountText", "videoCountText", "country", "channelId"):
            m = re.search(r'"%s":"((?:[^"\\]|\\.)*)"' % key, html)
            if m:
                about[key] = json.loads('"%s"' % m.group(1))
        m = re.search(r'"joinedDateText":\{"content":"([^"]+)"', html)
        if m:
            about["joinedDateText"] = {"content": m.group(1)}
    cid = (meta or {}).get("externalId") or about.get("channelId")
    if not cid:
        m = re.search(r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{22})"', html)
        cid = m.group(1) if m else None
    if not cid or not re.fullmatch(r"UC[\w-]{22}", cid):
        raise ParseError("no channel id in page")
    title = (meta or {}).get("title")
    if not title:
        m = re.search(r'<meta property="og:title" content="([^"]*)"', html)
        title = m.group(1) if m else None
    subs, _ = parse_count(about.get("subscriberCountText"))
    views, _ = parse_count(about.get("viewCountText"))
    vids, _ = parse_count(about.get("videoCountText"))
    joined_text = (about.get("joinedDateText") or {}).get("content") or ""
    joined = None
    m = re.search(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", joined_text)
    if m:
        try:
            joined = dt.datetime.strptime(" ".join(m.groups()), "%b %d %Y").date().isoformat()
        except ValueError:
            joined = None
    handle = None
    m = re.search(r"/@([^/?#]+)", about.get("canonicalChannelUrl") or (meta or {}).get("vanityChannelUrl") or "")
    if m:
        handle = urllib.parse.unquote(m.group(1))
    if views is None and subs is None:
        raise ParseError("no stats in About page")
    return {"channel_id": cid, "title": title, "handle_now": handle, "subscribers": subs,
            "total_views": views, "video_count": vids, "joined": joined, "country": about.get("country")}


def parse_rss(xml_text):
    """Latest uploads from the channel feed: exact views and publish times."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ParseError("bad RSS XML: %s" % e) from None
    if not root.tag.endswith("feed"):
        raise ParseError("not a feed")
    created = ts_of(root.findtext("a:published", default="", namespaces=NS))
    vids = []
    for e in root.findall("a:entry", NS):
        vid = e.findtext("yt:videoId", default="", namespaces=NS)
        if not vid:
            continue
        stats = e.find("media:group/media:community/media:statistics", NS)
        link = e.find("a:link[@rel='alternate']", NS)
        href = link.get("href", "") if link is not None else ""
        views = None
        if stats is not None and (stats.get("views") or "").isdigit():
            views = int(stats.get("views"))
        pub = ts_of(e.findtext("a:published", default="", namespaces=NS))
        vids.append({"id": vid, "title": (e.findtext("a:title", default="", namespaces=NS) or "").strip(),
                     "published": iso(pub) if pub else None, "published_basis": "exact" if pub else None,
                     "views": views, "views_step": 0, "duration_s": None,
                     "is_short": True if "/shorts/" in href else (False if "/watch" in href else None)})
    return vids, (iso(created) if created else None)


def parse_uploads_page(html, fetched_ts, limit=PAGE_VIDEOS):
    """Latest uploads from the uploads playlist page (fallback source).

    Views are rounded by YouTube. Channels that mix Shorts and long videos get a list
    with rounded age labels; Shorts-only channels get a Shorts grid with no ages at all.
    """
    data = initial_data(html)
    if data is None:
        raise ParseError("no ytInitialData in uploads page")
    vids = []

    def add(vid, title, views_text, age_text, duration, short=None):
        if not vid or any(v["id"] == vid for v in vids):
            return
        views, step = parse_count(views_text)
        age_h = parse_age_hours(age_text)
        if short is None and duration is not None:
            short = duration <= 180   # heuristic: the list layout does not say Short or not
        vids.append({"id": vid, "title": (title or "").strip(),
                     "published": iso(fetched_ts - age_h * 3600) if age_h is not None else None,
                     "published_basis": "label" if age_h is not None else None,
                     "views": views, "views_step": step, "duration_s": duration, "is_short": short})

    for lv in walk(data, "lockupViewModel"):          # list layout (mixed channels)
        md = (lv.get("metadata") or {}).get("lockupMetadataViewModel") or {}
        views_text = age_text = None
        for row in ((md.get("metadata") or {}).get("contentMetadataViewModel") or {}).get("metadataRows") or []:
            for part in row.get("metadataParts") or []:
                txt = ((part.get("text") or {}).get("content") or "").strip()
                label = (part.get("accessibilityLabel") or "").lower()
                if "view" in label or "view" in txt.lower():
                    views_text = txt
                elif "ago" in label or "ago" in txt.lower():
                    age_text = txt
        duration = None
        for badge in walk(lv.get("contentImage"), "thumbnailBadgeViewModel"):
            duration = parse_duration(badge.get("text") or "")
            if duration is not None:
                break
        add(lv.get("contentId"), (md.get("title") or {}).get("content"), views_text, age_text, duration)
    for sv in walk(data, "shortsLockupViewModel"):    # Shorts grid (Shorts-only channels): no ages
        cmd = (sv.get("onTap") or {}).get("innertubeCommand") or {}
        vid = (cmd.get("reelWatchEndpoint") or {}).get("videoId")
        if not vid:
            m = re.search(r"/shorts/([\w-]{11})", json.dumps(cmd))
            vid = m.group(1) if m else None
        meta = sv.get("overlayMetadata") or {}
        add(vid, (meta.get("primaryText") or {}).get("content"),
            (meta.get("secondaryText") or {}).get("content"), None, None, short=True)
    for pv in walk(data, "playlistVideoRenderer"):    # older list layout (kept as a safety net)
        info = "".join(r.get("text", "") for r in (pv.get("videoInfo") or {}).get("runs") or [])
        m = re.search(r"([\d.,]+\s*[KMB]?|No)\s+views?", info, re.I)
        title = "".join(r.get("text", "") for r in (pv.get("title") or {}).get("runs") or [])
        length = pv.get("lengthSeconds")
        add(pv.get("videoId"), title, m.group(1) if m else None, info.split("\u2022")[-1] if "ago" in info else None,
            int(length) if str(length or "").isdigit() else None)
    if not vids and ("alertRenderer" in html or "does not exist" in html):
        raise ParseError("uploads playlist unavailable")
    return vids[:limit]


# ----------------------------------------------------------------------------- files

def paths(base):
    return {"watchlist": os.path.join(base, "watchlist.json"), "data": os.path.join(base, "data"),
            "summary": os.path.join(base, "latest_summary.md")}


def write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_watchlist(p):
    with open(p["watchlist"], encoding="utf-8") as f:
        wl = json.load(f)
    if isinstance(wl, list):
        wl = {"channels": wl}
    wl.setdefault("channels", [])
    return wl


def save_watchlist(p, wl):
    write_text(p["watchlist"], json.dumps(wl, ensure_ascii=False, indent=1) + "\n")


def dump_snapshot(path, snap):
    """Valid JSON with one channel per line (small diffs, small files)."""
    head = [" %s: %s," % (json.dumps(k), json.dumps(v, ensure_ascii=False))
            for k, v in snap.items() if k != "channels"]
    rows = ["  " + json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in snap["channels"]]
    write_text(path, "{\n" + "\n".join(head) + '\n "channels": [\n' + ",\n".join(rows) + "\n ]\n}\n")


def load_snapshot(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def previous_snapshot_date(data_dir, date):
    dates = []
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.json", name)
            if m and m.group(1) < date:
                dates.append(m.group(1))
    return max(dates) if dates else None


# ----------------------------------------------------------------------------- snapshot

def fetch_parsed(net, url, parser, *args):
    """GET and parse; a page that arrives incomplete (parse error) is fetched once more."""
    try:
        return parser(net.get(url), *args)
    except ParseError:
        return parser(net.get(url), *args)


def snapshot_channel(net, ch, state):
    rec = {"handle": ch.get("handle"), "name": ch.get("name") or ch.get("handle"),
           "category": ch.get("category"), "counter_example": bool(ch.get("counter_example")),
           "channel_id": ch.get("channel_id"), "fetched_at": iso(time.time()),
           "subscribers": None, "total_views": None, "video_count": None, "joined": None,
           "country": None, "source": None, "videos": [], "error": None}
    cid = ch.get("channel_id")
    url = ABOUT_BY_ID.format(cid) if cid else ABOUT_BY_HANDLE.format(urllib.parse.quote(ch["handle"]))
    try:
        info = fetch_parsed(net, url, parse_about)
        if cid and info["channel_id"] != cid:
            raise ParseError("page returned a different channel id")
        if not cid:
            cid = rec["channel_id"] = ch["channel_id"] = info["channel_id"]
            state["watchlist_changed"] = True
        for key in ("subscribers", "total_views", "video_count", "joined", "country"):
            rec[key] = info[key]
        if info["title"]:
            rec["name"] = info["title"]
        if info["handle_now"] and info["handle_now"].lower() != str(ch.get("handle", "")).lower():
            rec["handle_now"] = info["handle_now"]
    except (FetchError, ParseError) as e:
        gone = " (channel removed or renamed?)" if getattr(e, "code", None) == 404 else ""
        rec["error"] = "about: %s%s" % (e, gone)
        if not cid:
            return rec
    notes = []
    if state["use_rss"]:
        try:
            rec["videos"], created = parse_rss(net.get(RSS_URL.format(cid), retries=0))
            rec["source"] = "rss"
            if created:
                rec["created"] = created
            state["rss_fail_streak"] = 0
        except (FetchError, ParseError) as e:
            notes.append("rss: %s" % e)
            state["rss_fail_streak"] += 1
            state["rss_errors"] += 1
            if state["rss_fail_streak"] >= RSS_FAILS_TO_DISABLE:
                state["use_rss"] = False     # feed looks down: stop wasting requests this run
    if rec["source"] is None and state["use_page"]:
        try:
            rec["videos"] = fetch_parsed(net, UPLOADS_URL.format("UU" + cid[2:]), parse_uploads_page, time.time())
            rec["source"] = "page"
        except (FetchError, ParseError) as e:
            notes.append("page: %s" % e)
    if rec["source"] is None:
        rec["video_error"] = "; ".join(notes) or "no video source enabled"
    return rec


def settle_publish_times(snap, prev):
    """Give every listed video the best available publish time and say where it came from.

    published_basis: "exact" (RSS), "label" (rounded age label on the uploads page), "seen"
    (first appeared between two snapshots, so the middle of that interval) or "order" (a guess
    from the position in the upload list and the channel's lifetime upload rate; heuristic).
    An estimate saved in an earlier snapshot is kept, because age labels get coarser with age.
    """
    prev_ch = {c.get("channel_id"): c for c in (prev or {}).get("channels", []) if c.get("channel_id")}
    for c in snap["channels"]:
        vids = c.get("videos") or []
        p = prev_ch.get(c.get("channel_id"))
        pv = {v["id"]: v for v in (p.get("videos") or [])} if p else {}
        first_known = next((i for i, v in enumerate(vids) if v["id"] in pv), None)
        t_now = ts_of(c.get("fetched_at")) or time.time()
        t_prev = ts_of(p.get("fetched_at")) if p else None
        per_upload = None   # days per upload, lifetime average
        if c.get("joined") and c.get("video_count"):
            per_upload = max((t_now - ts_of(c["joined"] + "T00:00:00Z")) / 86400, 1) / c["video_count"]
        for i, v in enumerate(vids):
            basis = v.get("published_basis")
            if basis == "exact":
                continue
            o = pv.get(v["id"])
            if o and o.get("published") and not (o.get("published_basis") == "order" and basis == "label"):
                v["published"], v["published_basis"] = o["published"], o.get("published_basis") or "label"
            elif basis == "label":
                continue
            elif first_known is not None and i < first_known and t_prev:
                v["published"], v["published_basis"] = iso((t_prev + t_now) / 2), "seen"
            elif per_upload:
                v["published"], v["published_basis"] = iso(t_now - (i + 0.5) * per_upload * 86400), "order"


def take_snapshot(p, date, source):
    wl = load_watchlist(p)
    started = time.time()
    net = Net(started + BUDGET_S)
    state = {"use_rss": source in ("auto", "rss"), "use_page": source in ("auto", "page"),
             "rss_fail_streak": 0, "rss_errors": 0, "watchlist_changed": False}
    path = os.path.join(p["data"], date + ".json")
    old = {}
    if os.path.exists(path):          # same-day rerun: keep good data for channels that fail now
        try:
            old = {c.get("handle"): c for c in load_snapshot(path).get("channels", [])}
        except (ValueError, OSError):
            old = {}
    channels = []
    for i, ch in enumerate(wl["channels"], 1):
        try:
            rec = snapshot_channel(net, ch, state)
        except Exception as e:   # one channel must never stop the run
            rec = {"handle": ch.get("handle"), "name": ch.get("name") or ch.get("handle"),
                   "category": ch.get("category"), "counter_example": bool(ch.get("counter_example")),
                   "channel_id": ch.get("channel_id"), "fetched_at": iso(time.time()), "videos": [],
                   "source": None, "error": "unexpected %s: %s" % (type(e).__name__, str(e)[:120])}
        prior = old.get(rec["handle"])
        if rec.get("error") and prior and not prior.get("error"):
            prior["carried_from_earlier_run"] = True
            prior["error_now"] = rec["error"]
            rec = prior
        channels.append(rec)
        flag = rec.get("error") or rec.get("video_error") or ""
        print("[%2d/%d] %-24s %-4s views=%s videos=%d %s" % (
            i, len(wl["channels"]), str(rec["handle"])[:24], rec.get("source") or "-",
            rec.get("total_views"), len(rec.get("videos") or []), flag), flush=True)
    if state["watchlist_changed"]:
        save_watchlist(p, wl)
    snap = {"date": date, "started_at": iso(started), "finished_at": iso(time.time()),
            "runtime_s": round(time.time() - started, 1), "requests": net.requests,
            "rss_errors": state["rss_errors"], "rss_disabled_during_run": not state["use_rss"] and source != "page",
            "notes": "views_step > 0 means YouTube rounded the count (true value assumed in [views, views+step)); "
                     "published_basis: exact = RSS, label = rounded age label, seen = first appeared between two "
                     "snapshots, order = guess from upload order and lifetime upload rate (heuristic)",
            "channels": channels}
    prev_date = previous_snapshot_date(p["data"], date)
    prev = load_snapshot(os.path.join(p["data"], prev_date + ".json")) if prev_date else None
    settle_publish_times(snap, prev)
    os.makedirs(p["data"], exist_ok=True)
    dump_snapshot(path, snap)
    return snap, prev


# ----------------------------------------------------------------------------- analysis

def est(v):
    """Best single estimate of a view count (middle of the rounding interval)."""
    return v["views"] + (v.get("views_step") or 0) / 2


def fmt(n, sign=False):
    if n is None:
        return "?"
    s = "+" if sign and n > 0 else ("-" if n < 0 else "")
    a = abs(n)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            x = a / div
            return "%s%s%s" % (s, ("%.2f" if x < 10 else "%.1f" if x < 100 else "%.0f") % x, suf)
    return "%s%d" % (s, round(a))


def median_latest(c, n=15):
    vals = [est(v) for v in (c.get("videos") or [])[:n] if v.get("views") is not None]
    return statistics.median(vals) if vals else None


def video_rows(snap, prev):
    """One row per listed video with `metric` = view gain since prev, or views/day on a first run."""
    now = ts_of(snap["started_at"])
    prev_ch = {c.get("channel_id"): c for c in (prev or {}).get("channels", []) if c.get("channel_id")}
    rows = []
    for c in snap["channels"]:
        vids = c.get("videos") or []
        p = prev_ch.get(c.get("channel_id"))
        pv = {v["id"]: v for v in (p.get("videos") or [])} if p else {}
        p_time = ts_of(p.get("fetched_at")) if p else None
        first_known = next((i for i, v in enumerate(vids) if v["id"] in pv), None)
        fetched = ts_of(c.get("fetched_at")) or now
        for i, v in enumerate(vids):
            if v.get("views") is None:
                continue
            pub = ts_of(v.get("published"))
            age_d = max((fetched - pub) / 86400, 0) if pub else None
            basis = v.get("published_basis")
            row = {"id": v["id"], "title": v.get("title") or "", "channel": c.get("name"),
                   "handle": c.get("handle"), "counter": c.get("counter_example"), "views": est(v),
                   "approx": bool(v.get("views_step")), "views_approx": bool(v.get("views_step")),
                   "age_d": age_d, "new": False, "age_mark": {"exact": "", "order": "≈"}.get(basis, "~"),
                   "long": v.get("is_short") is False, "metric": None}
            if prev is None:
                if age_d is not None:
                    # heuristic floor: 6 hours for exact ages, 1 day for estimated ones (avoids inflation)
                    row["metric"] = est(v) / max(age_d, 0.25 if basis == "exact" else 1.0)
                    row["approx"] = row["approx"] or basis != "exact"
            elif v["id"] in pv and pv[v["id"]].get("views") is not None:
                o = pv[v["id"]]
                if row["approx"] or o.get("views_step"):
                    same = v["views"] == o["views"] and (v.get("views_step") or 0) == (o.get("views_step") or 0)
                    row["metric"] = 0 if same else max(0, est(v) - est(o))
                    row["approx"] = True
                else:
                    row["metric"] = v["views"] - o["views"]
            elif p is not None:
                # listed above every video already known = uploaded since the previous snapshot
                is_new = (first_known is not None and i < first_known) or (
                    first_known is None and pub is not None and p_time is not None and pub > p_time)
                if is_new:
                    row["metric"], row["new"] = est(v), True
            rows.append(row)
    return rows


def channel_rows(snap, prev):
    now = ts_of(snap["started_at"])
    prev_ch = {c.get("channel_id"): c for c in (prev or {}).get("channels", []) if c.get("channel_id")}
    rows = []
    for c in snap["channels"]:
        p = prev_ch.get(c.get("channel_id"))
        tv, vc = c.get("total_views"), c.get("video_count")
        row = {"name": c.get("name"), "handle": c.get("handle"), "cat": c.get("category"),
               "counter": c.get("counter_example"), "total": tv, "subs": c.get("subscribers"), "videos": vc,
               "per_video": (tv / vc) if tv is not None and vc else None, "median": median_latest(c),
               "error": c.get("error"), "gain": None, "d_subs": None, "d_videos": None, "life_per_day": None}
        if p and tv is not None and p.get("total_views") is not None:
            row["gain"] = tv - p["total_views"]
        if p and c.get("subscribers") is not None and p.get("subscribers") is not None:
            row["d_subs"] = c["subscribers"] - p["subscribers"]
        if p and vc is not None and p.get("video_count") is not None:
            row["d_videos"] = vc - p["video_count"]
        joined = ts_of(c["joined"] + "T00:00:00Z") if c.get("joined") else None
        if joined and tv is not None:
            row["life_per_day"] = tv / max((now - joined) / 86400, 1)
        vids = c.get("videos") or []
        if not vids or any(v.get("published_basis") in (None, "order") for v in vids):
            row["uploads_7d"] = "?"     # no real dates for this list (never count guessed ages)
        else:
            week = [v for v in vids if now - ts_of(v["published"]) <= 7 * 86400]
            row["uploads_7d"] = "%d%s" % (len(week), "+" if week and len(week) == len(vids) else "")
        rows.append(row)
    return rows


def title_words(rows, top=40, per_channel=6):
    words, tags, used, seen = Counter(), Counter(), 0, Counter()
    for r in rows:
        if used >= top:
            break
        if seen[r["handle"]] >= per_channel:
            continue
        seen[r["handle"]] += 1
        used += 1
        toks = set(re.findall(r"#?\w+", r["title"].lower()))
        tags.update(t for t in toks if t.startswith("#") and t not in GENERIC_TAGS)
        words.update(t for t in toks if not t.startswith("#") and len(t) > 1 and not t.isdigit() and t not in STOP)
    return words, tags, used


def cut(text, n):
    text = re.sub(r"\s+", " ", text or "").replace("|", "/").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build_summary(snap, prev):
    first = prev is None
    # videos need an earlier video list to have a gain; without one they are ranked by views per day
    vfirst = first or not any(c.get("videos") for c in prev.get("channels", []))
    vrows = [r for r in video_rows(snap, None if vfirst else prev) if r["metric"] is not None]
    vrows.sort(key=lambda r: r["metric"], reverse=True)
    crows = channel_rows(snap, prev)
    key = "life_per_day" if first else "gain"
    crows.sort(key=lambda r: (r[key] is not None, r[key] or 0), reverse=True)
    chans = snap["channels"]
    ok = sum(1 for c in chans if not c.get("error"))
    n_rss = sum(1 for c in chans if c.get("source") == "rss")
    n_page = sum(1 for c in chans if c.get("source") == "page")
    n_none = len(chans) - n_rss - n_page
    L = ["# Competitor tracker %s" % snap["date"]]
    if first:
        L.append("Baseline run (no earlier snapshot): videos ranked by views per day since publication, channels by "
                 "lifetime views per day since the join date. Both are heuristics, not daily gains; real gains "
                 "start with the second snapshot.")
    else:
        hours = (ts_of(snap["started_at"]) - ts_of(prev["started_at"])) / 3600
        L.append("Compared with %s (%.1f hours earlier). Channel gain = change in total channel views (exact counts, "
                 "but YouTube updates them in steps, so a single day's gain is noisy). %s%s" % (
                     prev["date"], hours,
                     "The earlier snapshot has no video lists, so videos are ranked by views per day since "
                     "publication (heuristic)." if vfirst else
                     "Video gain = change in views of the listed latest uploads.",
                     " Note: " + prev["seed_note"] if prev.get("seed_note") else ""))
    L.append("Channels: %d tracked, %d with stats. Video lists: %d from RSS (exact), %d from the uploads page, "
             "%d unavailable. Marks: ~ rounded by YouTube; ≈ age guessed from upload order (heuristic, used when "
             "the page shows no dates). Run took %ss, %s requests." % (
                 len(chans), ok, n_rss, n_page, n_none, snap.get("runtime_s"), snap.get("requests")))
    L.append("")
    L.append("## Top 20 videos by %s" % ("views per day since publication" if vfirst else "view gain since previous snapshot"))
    L.append("| # | %s | views | age | channel | title | id |" % ("views/day" if vfirst else "gain"))
    L.append("|---|---|---|---|---|---|---|")
    top, left_out, per = [], Counter(), Counter()
    for r in vrows:
        if len(top) >= 20:
            break
        if per[r["handle"]] >= TOP_PER_CHANNEL:
            left_out[r["channel"]] += 1
            continue
        per[r["handle"]] += 1
        top.append(r)
    for i, r in enumerate(top, 1):
        age = "?" if r["age_d"] is None else r["age_mark"] + (
            "%.0fh" % (r["age_d"] * 24) if r["age_d"] < 2 else "%.0fd" % r["age_d"])
        mark = "≈" if vfirst and r["age_mark"] == "≈" else ("~" if r["approx"] else "")
        L.append("| %d | %s%s%s | %s%s | %s | %s%s | %s%s | %s |" % (
            i, mark, fmt(r["metric"], sign=not vfirst), " new" if r["new"] else "",
            "~" if r["views_approx"] else "", fmt(r["views"]), age, cut(r["channel"], 22),
            " (counter-example)" if r["counter"] else "", cut(r["title"], 62), " [long]" if r["long"] else "", r["id"]))
    if not vrows:
        L.append("| - | no video data | | | | | |")
    if left_out:
        L.append("At most %d rows per channel are shown. Higher-ranked rows left out: %s." % (
            TOP_PER_CHANNEL, ", ".join("%s %d" % kv for kv in left_out.most_common())))
    L.append("")

    def chan_line(i, r):
        gain = fmt(r["life_per_day"]) if first else fmt(r["gain"], sign=True)
        subs = fmt(r["subs"]) + ("" if first or r["d_subs"] is None else " (%s)" % (fmt(r["d_subs"], sign=True) if r["d_subs"] else "0"))
        vids = ("?" if r["videos"] is None else str(r["videos"])) + (
            "" if first or r["d_videos"] is None else " (%+d)" % r["d_videos"])
        return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            i, cut(r["name"], 24), r["cat"] or "", gain, fmt(r["total"]), subs, vids,
            fmt(r["per_video"]), fmt(r["median"]), r["uploads_7d"])

    head = "| # | channel | type | %s | total views | subs | videos | views per video | median latest 15 | uploads 7d |" % (
        "views/day (lifetime)" if first else "gain")
    main = [r for r in crows if not r["counter"]]
    counters = [r for r in crows if r["counter"]]
    L.append("## Channels ranked by %s" % ("lifetime views per day" if first else "total-view gain"))
    L.append(head)
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    chan_at = len(L)
    shown = main[:12]
    L.extend(chan_line(i, r) for i, r in enumerate(shown, 1))
    rest = main[len(shown):]
    rest_at = len(L)
    L.append("")   # placeholder, filled after trimming
    L.append("")
    L.append("## Counter-examples (failed channels kept on purpose)")
    L.append(head)
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    L.extend(chan_line(i, r) for i, r in enumerate(counters[:10], 1))
    tot_c = sum(r["total"] or 0 for r in counters)
    vid_c = sum(r["videos"] or 0 for r in counters)
    tot_m = sum(r["total"] or 0 for r in main)
    vid_m = sum(r["videos"] or 0 for r in main)
    agg = "All-time views per uploaded video: counter-examples %s (%d channels), other tracked channels %s (%d channels)." % (
        fmt(tot_c / vid_c) if vid_c else "?", len(counters), fmt(tot_m / vid_m) if vid_m else "?", len(main))
    if not first:
        gc = sum(r["gain"] or 0 for r in counters)
        gm = sum(r["gain"] or 0 for r in main)
        agg += " Gain since previous snapshot: counter-examples %s in total, others %s." % (fmt(gc, True), fmt(gm, True))
    L.append(agg)
    L.append("")
    words, tags, used = title_words(vrows)
    L.append("## Most common title words (top %d videos above, at most 6 per channel; count = titles containing the word)" % used)
    L.append("Words: " + (", ".join("%s %d" % kv for kv in words.most_common(18)) or "none"))
    L.append("Hashtags: " + (", ".join("%s %d" % kv for kv in tags.most_common(10)) or "none"))
    problems = []
    if snap.get("rss_errors"):
        problems.append("RSS feed failed %d time(s)%s; the uploads page was used instead (rounded counts)." % (
            snap["rss_errors"], ", then was skipped for the rest of the run" if snap.get("rss_disabled_during_run") else ""))
    bad = ["%s (%s)" % (c.get("handle"), cut(c.get("error") or c.get("video_error"), 40))
           for c in chans if c.get("error") or c.get("video_error")]
    if bad:
        problems.append("Channel errors: " + "; ".join(bad[:8]) + (" and %d more" % (len(bad) - 8) if len(bad) > 8 else ""))
    carried = [c.get("handle") for c in chans if c.get("carried_from_earlier_run")]
    if carried:
        problems.append("Kept data from an earlier run today for: " + ", ".join(carried[:10]))
    if problems:
        L.append("")
        L.append("## Problems")
        L.extend("- " + x for x in problems[:3])
    # enforce the line limit by moving channels from the table into the one-line remainder
    while len(L) > SUMMARY_MAX_LINES and len(shown) > 3:
        rest.insert(0, shown.pop())
        del L[chan_at + len(shown)]
        rest_at -= 1
    L[rest_at] = ("Others: " + ", ".join("%s %s" % (cut(r["name"], 20), fmt(r["life_per_day"]) if first else fmt(r["gain"], True))
                                           for r in rest)) if rest else "Others: none"
    return "\n".join(L[:SUMMARY_MAX_LINES]) + "\n"


# ----------------------------------------------------------------------------- commands

def add_channels(p, handles, category, counter):
    wl = load_watchlist(p)
    net = Net(time.time() + BUDGET_S)
    rc = 0
    for raw in handles:
        m = re.search(r"@([^/?#\s]+)", raw)
        handle = urllib.parse.unquote(m.group(1) if m else raw.strip().strip("/").split("/")[-1])
        if any(str(c.get("handle", "")).lower() == handle.lower() for c in wl["channels"]):
            print("already in watchlist: @%s" % handle)
            continue
        try:
            info = parse_about(net.get(ABOUT_BY_HANDLE.format(urllib.parse.quote(handle))))
        except (FetchError, ParseError) as e:
            print("NOT added @%s: %s" % (handle, e))
            rc = 1
            continue
        if any(c.get("channel_id") == info["channel_id"] for c in wl["channels"]):
            print("already in watchlist (same channel id): @%s" % handle)
            continue
        wl["channels"].append({"handle": info["handle_now"] or handle, "name": info["title"] or handle,
                               "category": category, "counter_example": counter,
                               "channel_id": info["channel_id"], "format": None, "notes": "",
                               "source": "added %s with --add" % dt.datetime.now(dt.timezone.utc).date().isoformat()})
        print("added @%s  %s  %s  subs=%s views=%s videos=%s" % (
            handle, info["channel_id"], info["title"], info["subscribers"], info["total_views"], info["video_count"]))
    save_watchlist(p, wl)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily competitor snapshot for code-made Shorts channels.")
    ap.add_argument("--date", help="snapshot date label YYYY-MM-DD (default: today, UTC)")
    ap.add_argument("--add", action="append", metavar="HANDLE", help="append a channel to the watchlist and exit (repeatable)")
    ap.add_argument("--category", default="sim", help="category stored with --add (default: sim)")
    ap.add_argument("--counter-example", action="store_true", help="mark the channel added with --add as a counter-example")
    ap.add_argument("--summary-only", action="store_true", help="rebuild latest_summary.md from saved snapshots, no network")
    ap.add_argument("--source", choices=("auto", "rss", "page"), default="auto",
                    help="video list source: auto = RSS, falling back to the uploads page (default)")
    ap.add_argument("--dir", default=HERE, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    p = paths(os.path.abspath(args.dir))
    if args.add:
        return add_channels(p, args.add, args.category, args.counter_example)
    date = args.date or dt.datetime.now(dt.timezone.utc).date().isoformat()
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError
        dt.date.fromisoformat(date)
    except ValueError:
        ap.error("--date must be a real date written YYYY-MM-DD")
    t0 = time.time()
    if args.summary_only:
        path = os.path.join(p["data"], date + ".json")
        if not os.path.exists(path):
            print("no snapshot for %s at %s" % (date, path))
            return 1
        snap = load_snapshot(path)
        prev_date = previous_snapshot_date(p["data"], date)
        prev = load_snapshot(os.path.join(p["data"], prev_date + ".json")) if prev_date else None
    else:
        snap, prev = take_snapshot(p, date, args.source)
    write_text(p["summary"], build_summary(snap, prev))
    failed = sum(1 for c in snap["channels"] if c.get("error"))
    print("done in %.0fs: %d channels (%d failed), snapshot data/%s.json, summary latest_summary.md%s" % (
        time.time() - t0, len(snap["channels"]), failed, date,
        "" if prev else " (baseline: no earlier snapshot)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
