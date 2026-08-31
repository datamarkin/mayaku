"""Harvest a permissively-licensed, high-resolution photo corpus from Wikimedia Commons.

Purpose
-------
Build a raw image pool that can later be *labelled* into an Objects365-style
detection set: everyday scenes containing countable objects, at high enough
resolution that small objects survive annotation, under licenses whose
obligations an Apache-2.0 model release can actually satisfy.

Commons is ~120M files, but the fraction that is useful here is much smaller.
Three filters do the work, in this order:

1. **License** (client-side, from ``extmetadata``). Public domain, CC0, CC BY
   and CC BY-SA are accepted. NonCommercial, NoDerivs, GFDL-only, and anything
   carrying a ``Restrictions`` flag are dropped. See ``LICENSE POLICY`` below.
2. **Resolution.** ``--min-side`` is a *floor*, default 1500px on the shortest
   side; by default the full-resolution original is downloaded. Search-mode
   entries push the floor server-side via ``filew:``/``fileh:``; category-mode
   entries filter client-side from ``imageinfo``. Either way the downloaded
   pixels are re-checked, because Commons' recorded dimensions are sometimes
   wrong.
3. **Content** (the query plan). Commons' bulk is maps, scanned books, coats of
   arms, herbarium sheets and building facades — none of which resemble
   Objects365. The plan steers the crawl at photographic subject categories
   instead of taking the corpus as it comes.

Politeness
----------
Hard-coded to the Wikimedia robot policy (https://wikitech.wikimedia.org/wiki/Robot_policy):
at most 2 concurrent connections to ``upload.wikimedia.org``, a total download
ceiling of 20 Mbps (policy allows 25), serial metadata requests, ``Retry-After``
honoured on 429, and a 15-minute global pause on any 5xx. A contact-bearing
User-Agent is *required* — the script refuses to start without one, because
generic agents get IP-blocked and deserve to be.

Under those limits a single external IP sustains roughly 8-12 images/sec, i.e.
~1M images/day. If you need more than that, run it inside Toolforge/WMCS
(exempt from rate limits) or email the WMF traffic team first.

LICENSE POLICY
--------------
Accepted:  CC0, public domain (any PD-* tag), CC BY 1.0-4.0, CC BY-SA 1.0-4.0.
Rejected:  any NC / ND, GFDL-only, Free Art License, and any file with a
           non-empty ``Restrictions`` field (trademark, personality rights,
           insignia).

Share-alike is included on purpose — it is the most common license on Commons
and excluding it costs well over half the corpus. Reversing that is a two-token
edit at ``_SLUG_DENY`` / ``_TEXT_DENY``, documented there.

Both CC BY and CC BY-SA require attribution. ``manifest.jsonl`` records the
author, credit line, license and source URL for every accepted file — keep it
with the dataset.

Usage
-----
Dump the default plan, edit it, then check what it would actually yield::

    python tools/harvest_commons.py --dump-plan plan.json
    python tools/harvest_commons.py --plan plan.json --estimate \\
        --user-agent "mayaku-harvest/1.0 (https://github.com/datamarkin/mayaku; you@example.com)"

Then crawl::

    python tools/harvest_commons.py --plan plan.json --out /data/commons \\
        --min-side 1500 --fetch original \\
        --user-agent "mayaku-harvest/1.0 (https://github.com/datamarkin/mayaku; you@example.com)"

Resumable: state lives in ``<out>/state.db``, re-running continues where it
stopped and never re-downloads or re-considers a file it has already judged.

Output layout::

    <out>/
    ├── images/ab/cd/<sha1>.jpg     # sharded by sha1 prefix
    ├── manifest.jsonl              # one record per accepted image
    └── state.db                    # sqlite resume + dedup state

Sizing. Originals of qualifying files measure ~4.1MB median / ~5.0MB mean, at a
median 12MP (sampled over three categories). That is ~10TB and, at the 20 Mbps
policy ceiling, ~46 days of download for a 2M-image pool from one external IP —
bandwidth, not the API, is the binding constraint.

Two ways to spend less, both optional and both lossy:
  ``--fetch thumb``   take the smallest permitted thumbnail still clearing
                      --min-side. ~4x fewer bytes; caps resolution at the floor.
  ``--resize-to PX``  keep originals but downscale on arrival, to trade CPU and
                      resolution for disk while paying full download cost.
Neither is on by default. Wikimedia serves only a fixed ladder of thumbnail
widths (960/1280/1920/3840), so ``--fetch thumb`` on a 3:2 landscape photo pulls
3840px to retain 1500px of height.

Dependencies: stdlib only. Pillow, if importable, is used to verify decodes and
reject animations — strongly recommended, and required for ``--resize-to``.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia *enforces* a fixed set of thumbnail widths on direct requests to
# upload.wikimedia.org — anything else is rejected with HTTP 400/429 rather than
# rendered (https://www.mediawiki.org/wiki/Common_thumbnail_sizes, T414805).
# Only the API round-trip rounds up for you; constructed URLs must already
# comply, so the ladder is a subset of the permitted list, not a guess.
STANDARD_THUMB_WIDTHS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)
THUMB_LADDER = tuple(w for w in STANDARD_THUMB_WIDTHS if w >= 960)

# Only formats that thumbnail to themselves under the /thumb/ URL scheme.
# GIF is excluded as an animation format; TIFF renders to `lossy-page1-*.jpg`
# and is nearly always a scan anyway; SVG is a vector and excluded by
# `filetype:bitmap` server-side as well.
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

EXTMETA_KEYS = (
    "License|LicenseShortName|UsageTerms|LicenseUrl|Artist|Credit|"
    "Restrictions|ObjectName|DateTimeOriginal"
)

# --------------------------------------------------------------------------
# license classification
# --------------------------------------------------------------------------

# `extmetadata.License` is a machine-readable slug (`cc0`, `pd`, `cc-by-4.0`,
# `cc-by-sa-3.0`, `gfdl`, ...). Match on that, not on the prose in UsageTerms —
# most PD templates say "...because its copyright has expired", and a substring
# hunt for "copyright" across free text throws away the whole public domain.
# Share-alike is accepted, deliberately: CC BY-SA is by a wide margin the most
# common license on Commons, and rejecting it costs well over half the corpus.
# Whether training weights on SA-licensed images makes those weights a derivative
# work is unsettled rather than settled against it. To reverse this, put `sa`
# back in _SLUG_DENY and `share.?alike` back in _TEXT_DENY.
_SLUG_DENY = re.compile(r"(^|[-_])(nc|nd|gfdl|fdl|fal|nolicense)($|[-_.])", re.I)
_SLUG_ALLOW = re.compile(
    r"^(cc0|cc.?zero|pd([-_.]|$)|public.?domain|cc.?by(.?sa)?([-_.][1-4]|$)|attribution$)",
    re.I,
)
# Belt-and-braces on the human-readable name, for files whose slug is missing or
# unrecognised upstream. These strings are stable across the license templates.
# GFDL is not listed here: a GFDL-only file carries the `gfdl` slug and is caught
# above, whereas the common "GFDL or CC BY-SA" dual grant is fine to take.
_TEXT_DENY = re.compile(
    r"non.?commercial|no.?deriv|free art licen|fair use|non.?free|all rights reserved",
    re.I,
)


def classify_license(extmeta: dict[str, Any]) -> tuple[bool, str, dict[str, str]]:
    """Return (accepted, normalised-slug-or-reason, attribution fields)."""

    def val(key: str) -> str:
        entry = extmeta.get(key)
        if isinstance(entry, dict):
            return str(entry.get("value", "") or "")
        return ""

    lic = val("License").strip()
    short = val("LicenseShortName").strip()
    terms = val("UsageTerms").strip()
    restrictions = val("Restrictions").strip()

    attribution = {
        "license": lic or short,
        "license_short": short,
        "license_url": val("LicenseUrl"),
        "artist": _strip_html(val("Artist")),
        "credit": _strip_html(val("Credit")),
        "title": _strip_html(val("ObjectName")),
    }

    # Trademark / personality-rights / insignia flags. Conservative: a file that
    # needs a second permission on top of its license is not worth the risk.
    if restrictions:
        return False, f"restricted:{restrictions[:40]}", attribution

    slug = (lic or short).strip()
    if not slug:
        return False, "license:missing", attribution

    if _TEXT_DENY.search(f"{short} {terms}"):
        return False, f"license:denied:{slug[:40]}", attribution
    if _SLUG_DENY.search(slug):
        return False, f"license:denied:{slug[:40]}", attribution
    if not _SLUG_ALLOW.search(slug):
        return False, f"license:unknown:{slug[:40]}", attribution
    return True, slug, attribution


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#039;", "'")
        .replace("&nbsp;", " ")
    )
    return _WS.sub(" ", text).strip()[:500]


# --------------------------------------------------------------------------
# content filters
# --------------------------------------------------------------------------

# Commons is full of high-resolution material that is useless as detection
# pretraining data. Filename patterns catch most of it more reliably than
# category negation does, because these files are consistently named.
_JUNK_NAME = re.compile(
    r"\b(map|karte|mapa|carte|plan|blueprint|diagram|chart|graph|schema|schematic"
    r"|coat.?of.?arms|wappen|escudo|blason|crest|seal|flag|banner|emblem|logo"
    r"|stamp|briefmarke|banknote|coin|medal|herbarium|specimen|microscop"
    r"|scan|scanned|page.?\d+|folio|manuscript|charter|census|register"
    r"|screenshot|spectrum|histogram|timeline|panorama.?stitch"
    r"|satellite|orthophoto|aerial.?view|topograph|nautical.?chart)\b",
    re.I,
)


def looks_like_junk(name: str) -> bool:
    return bool(_JUNK_NAME.search(name.replace("_", " ")))


# --------------------------------------------------------------------------
# default plan
# --------------------------------------------------------------------------

# Categories chosen for Objects365-like content: things a person would draw a
# box around. Depth is deliberately shallow — the Commons category graph drifts
# badly past 3 hops (Category:Food reaches Category:Agriculture reaches
# Category:Maps of agriculture). Run --estimate before trusting any of these;
# some will be thin, and the estimate tells you which.
DEFAULT_CATEGORIES: tuple[tuple[str, int], ...] = (
    # animals
    ("Category:Cats", 3),
    ("Category:Dogs", 3),
    ("Category:Horses", 3),
    ("Category:Cattle", 3),
    ("Category:Sheep", 3),
    ("Category:Birds", 2),
    ("Category:Insects", 2),
    ("Category:Fish", 2),
    # people in scenes
    ("Category:Pedestrians", 3),
    ("Category:Crowds", 3),
    ("Category:Children playing", 3),
    ("Category:People at work", 3),
    ("Category:People eating", 3),
    ("Category:Cyclists", 3),
    # vehicles
    ("Category:Automobiles", 2),
    ("Category:Buses", 2),
    ("Category:Trucks", 2),
    ("Category:Motorcycles", 3),
    ("Category:Bicycles", 3),
    ("Category:Trains", 2),
    ("Category:Aircraft", 2),
    ("Category:Boats", 2),
    ("Category:Ships", 2),
    ("Category:Tractors", 3),
    # street / outdoor scenes
    ("Category:Streets", 2),
    ("Category:Markets", 3),
    ("Category:Street markets", 3),
    ("Category:Shops", 3),
    ("Category:Restaurants", 3),
    ("Category:Cafés", 3),
    ("Category:Playgrounds", 3),
    ("Category:Parks", 2),
    ("Category:Beaches", 2),
    ("Category:Traffic signs", 3),
    ("Category:Traffic lights", 3),
    ("Category:Benches", 3),
    ("Category:Construction sites", 3),
    ("Category:Farms", 3),
    ("Category:Gardens", 2),
    ("Category:Bridges", 2),
    # indoor scenes
    ("Category:Kitchens", 3),
    ("Category:Bathrooms", 3),
    ("Category:Bedrooms", 3),
    ("Category:Living rooms", 3),
    ("Category:Offices", 3),
    ("Category:Classrooms", 3),
    ("Category:Supermarkets", 3),
    # objects
    ("Category:Furniture", 3),
    ("Category:Chairs", 3),
    ("Category:Tables", 3),
    ("Category:Cookware", 3),
    ("Category:Tableware", 3),
    ("Category:Bottles", 3),
    ("Category:Cups", 3),
    ("Category:Hand tools", 3),
    ("Category:Musical instruments", 3),
    ("Category:Computers", 3),
    ("Category:Mobile phones", 3),
    ("Category:Cameras", 3),
    ("Category:Clocks", 3),
    ("Category:Lamps", 3),
    ("Category:Toys", 3),
    ("Category:Books", 3),
    ("Category:Clothing", 3),
    ("Category:Shoes", 3),
    ("Category:Hats", 3),
    ("Category:Bags", 3),
    ("Category:Umbrellas", 3),
    # food
    ("Category:Food", 2),
    ("Category:Fruit", 3),
    ("Category:Vegetables", 3),
    ("Category:Beverages", 3),
    ("Category:Bread", 3),
    ("Category:Cakes", 3),
    # sport
    ("Category:Association football", 2),
    ("Category:Basketball", 2),
    ("Category:Tennis", 2),
    ("Category:Skiing", 2),
    ("Category:Surfing", 2),
    ("Category:Running", 2),
)


def default_plan(target_per_entry: int) -> list[dict[str, Any]]:
    plan = [
        {
            "name": cat.split(":", 1)[1].lower().replace(" ", "_"),
            "mode": "category",
            "category": cat,
            "depth": depth,
            "target": target_per_entry,
        }
        for cat, depth in DEFAULT_CATEGORIES
    ]
    # Two search entries as a template for precision queries. `haswbstatement`
    # hits Structured Data on Commons, which is cleaner than category
    # membership where it exists — coverage is partial, so it supplements
    # rather than replaces the category walk.
    plan.append(
        {
            "name": "sdc_depicts_dog",
            "mode": "search",
            "search": "haswbstatement:P180=Q144",
            "target": target_per_entry,
        }
    )
    plan.append(
        {
            "name": "quality_images",
            "mode": "search",
            "search": 'incategory:"Quality images"',
            "target": target_per_entry * 4,
        }
    )
    return plan


# --------------------------------------------------------------------------
# http plumbing
# --------------------------------------------------------------------------


class GlobalPause:
    """A 5xx anywhere stops the whole crawl, per the robot policy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0

    def hold(self, seconds: float, why: str) -> None:
        with self._lock:
            new_until = max(self._until, time.time() + seconds)
            if new_until > self._until:
                self._until = new_until
                print(f"[pause] {why}: holding {seconds:.0f}s", file=sys.stderr)

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5.0))


class ByteBudget:
    """Token bucket over bytes, shared by the download threads."""

    def __init__(self, mbps: float) -> None:
        self.rate = mbps * 1_000_000 / 8.0
        self.capacity = self.rate * 2.0
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def take(self, nbytes: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= nbytes:
                    self._tokens -= nbytes
                    return
                deficit = (nbytes - self._tokens) / self.rate
            time.sleep(min(deficit, 1.0))


@dataclass
class Http:
    user_agent: str
    pause: GlobalPause
    token: str | None = None
    # 200 req/min is the 2026 global limit for a compliant-User-Agent client
    # (https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits). 0.32s leaves
    # a little headroom under it; a bot-flagged or high-limits account can go
    # faster, but nothing here depends on that.
    api_interval: float = 0.32

    def __post_init__(self) -> None:
        self._api_lock = threading.Lock()
        self._api_last = 0.0

    def _headers(self, api: bool) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip"}
        if api and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _open(self, url: str, api: bool, timeout: float):
        request = urllib.request.Request(url, headers=self._headers(api))
        return urllib.request.urlopen(request, timeout=timeout)

    def _retry_loop(self, url: str, api: bool, timeout: float, attempts: int):
        for attempt in range(attempts):
            self.pause.wait()
            try:
                return self._open(url, api, timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    delay = _retry_after(exc) or min(60.0, 2.0**attempt)
                    self.pause.hold(delay, "429 Too Many Requests")
                    continue
                if 500 <= exc.code < 600:
                    self.pause.hold(900.0, f"HTTP {exc.code}")
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # DNS blips, dropped sockets, laptop lid closing. On a crawl
                # measured in days these are certainties, not edge cases, so
                # back every thread off together rather than racing to retry.
                if attempt == attempts - 1:
                    raise
                self.pause.hold(min(300.0, 5.0 * 2.0**attempt), f"network: {exc}")
        raise RuntimeError(f"gave up after {attempts} attempts: {url}")

    def api(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {"format": "json", "formatversion": "2", **params}
        url = f"{API}?{urllib.parse.urlencode(params)}"
        with self._api_lock:
            gap = self.api_interval - (time.monotonic() - self._api_last)
            if gap > 0:
                time.sleep(gap)
            response = self._retry_loop(url, api=True, timeout=60.0, attempts=8)
            self._api_last = time.monotonic()
        with response:
            body = _read_body(response)
        payload = json.loads(body)
        if "error" in payload:
            raise ApiError(payload["error"].get("code", "?"), str(payload["error"]))
        return payload

    def fetch(self, url: str, budget: ByteBudget, max_bytes: int) -> bytes:
        response = self._retry_loop(url, api=False, timeout=120.0, attempts=5)
        with response:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                budget.take(len(chunk))
                total += len(chunk)
                if total > max_bytes:
                    raise TooBig(total)
                chunks.append(chunk)
        return b"".join(chunks)


class ApiError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TooBig(Exception):
    pass


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return 60.0


def _read_body(response) -> bytes:
    data = response.read()
    if response.headers.get("Content-Encoding") == "gzip":
        import gzip

        data = gzip.decompress(data)
    return data


# --------------------------------------------------------------------------
# thumbnail selection
# --------------------------------------------------------------------------


def pick_url(
    original: str, width: int, height: int, filesize: int, policy: Policy
) -> tuple[str, int]:
    """Choose what to actually download. Returns (url, requested_thumb_width).

    A requested width of 0 means the original file is fetched as-is.

    ``fetch="original"`` keeps full resolution, which is what you want when the
    pool will be labelled and ``min_side`` is a floor rather than a target. The
    one exception is a file above ``max_bytes``: Commons holds 100MP panoramas
    and gigapixel scans, and those step down to the largest permitted thumbnail
    that still clears ``min_side`` instead of being skipped outright.

    ``fetch="thumb"`` takes the *smallest* permitted width that still clears
    ``min_side``. That is roughly 4x cheaper in bytes and is the right mode if
    bandwidth, not resolution, is the binding constraint.
    """
    original = original.split("?", 1)[0]  # imageinfo appends utm_* tracking params
    short = min(width, height)

    if policy.fetch == "original":
        if filesize <= 0 or filesize <= policy.max_bytes:
            return original, 0
        # Oversized: step down to the widest permitted thumbnail that still
        # holds min_side. Largest first, so we lose as little as possible.
        for candidate in reversed(THUMB_LADDER):
            if candidate < width and short * candidate / width >= policy.min_side:
                return thumb_url(original, candidate), candidate
        return original, 0  # nothing smaller qualifies; max_bytes will skip it

    if short <= policy.min_side:
        return original, 0  # already at the floor; a thumbnail would only shrink it
    # Scaling to width W leaves a shortest side of short * W / width.
    needed = math.ceil(policy.min_side * width / short)
    for candidate in THUMB_LADDER:
        if needed <= candidate < width:
            return thumb_url(original, candidate), candidate
    return original, 0


def thumb_url(original: str, width: int) -> str:
    """`/commons/a/ab/X.jpg` -> `/commons/thumb/a/ab/X.jpg/<w>px-X.jpg`."""
    if width not in STANDARD_THUMB_WIDTHS:
        raise ValueError(f"{width}px is not a permitted thumbnail width")
    original = original.split("?", 1)[0]
    marker = "/commons/"
    index = original.find(marker)
    if index < 0:
        return original
    head = original[: index + len(marker)]
    tail = original[index + len(marker) :]
    basename = tail.rsplit("/", 1)[-1]
    return f"{head}thumb/{tail}/{width}px-{basename}"


# --------------------------------------------------------------------------
# image verification
# --------------------------------------------------------------------------


def downscale(data: bytes, short_target: int, quality: int) -> tuple[bytes, tuple[int, int]]:
    """Re-encode so the shortest side is exactly ``short_target``.

    The permitted thumbnail widths are coarse (960/1280/1920/3840), so a 3:2
    landscape photo has to come down at 3840px to keep 1500px of height —
    ~2.3MB where 1500px-short is ~600KB. Over a 2M-image pool that is the
    difference between ~4.6TB and ~1.2TB on disk, for pixels the target
    resolution does not use.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = short_target / min(width, height)
        if scale >= 1.0:
            return data, (width, height)
        size = (round(width * scale), round(height * scale))
        resized = image.resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue(), size


def verify(data: bytes, min_side: int) -> tuple[bool, str, tuple[int, int] | None]:
    try:
        from PIL import Image
    except ImportError:
        return True, "unverified", None

    import io

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            size = image.size
            frames = getattr(image, "n_frames", 1)
            mode = image.mode
    except Exception as exc:
        return False, f"decode:{type(exc).__name__}", None

    if frames > 1:
        return False, "animated", size
    if mode in {"1", "L", "LA"}:
        return False, "grayscale", size
    if min(size) < min_side:
        return False, f"small:{size[0]}x{size[1]}", size
    return True, "ok", size


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


class State:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                name    TEXT PRIMARY KEY,
                sha1    TEXT,
                status  TEXT NOT NULL,
                reason  TEXT,
                entry   TEXT,
                width   INTEGER,
                height  INTEGER,
                path    TEXT
            );
            CREATE INDEX IF NOT EXISTS files_sha1   ON files(sha1);
            CREATE INDEX IF NOT EXISTS files_status ON files(status);
            CREATE TABLE IF NOT EXISTS entries (
                name TEXT PRIMARY KEY,
                accepted INTEGER NOT NULL DEFAULT 0,
                done INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.conn.commit()
        self._lock = threading.Lock()

    def seen(self, name: str) -> bool:
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM files WHERE name = ?", (name,)).fetchone()
        return row is not None

    def sha1_taken(self, sha1: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM files WHERE sha1 = ? AND status = 'done'", (sha1,)
            ).fetchone()
        return row is not None

    def record(self, **kw: Any) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO files "
                "(name, sha1, status, reason, entry, width, height, path) "
                "VALUES (:name, :sha1, :status, :reason, :entry, :width, :height, :path)",
                {
                    "sha1": None,
                    "reason": None,
                    "width": None,
                    "height": None,
                    "path": None,
                    **kw,
                },
            )
            self.conn.commit()

    def entry_progress(self, name: str) -> tuple[int, bool]:
        with self._lock:
            row = self.conn.execute(
                "SELECT accepted, done FROM entries WHERE name = ?", (name,)
            ).fetchone()
        return (0, False) if row is None else (row[0], bool(row[1]))

    def bump_entry(self, name: str, accepted: int = 0, done: bool | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO entries(name, accepted, done) VALUES (?, 0, 0) "
                "ON CONFLICT(name) DO NOTHING",
                (name,),
            )
            if accepted:
                self.conn.execute(
                    "UPDATE entries SET accepted = accepted + ? WHERE name = ?",
                    (accepted, name),
                )
            if done is not None:
                self.conn.execute("UPDATE entries SET done = ? WHERE name = ?", (int(done), name))
            self.conn.commit()

    def totals(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT status, COUNT(*) FROM files GROUP BY status"
            ).fetchall()
        return dict(rows)


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------


def _imageinfo_params() -> dict[str, Any]:
    return {
        "prop": "imageinfo",
        "iiprop": "url|size|sha1|mime|extmetadata",
        "iiextmetadatafilter": EXTMETA_KEYS,
    }


def subcategories(http: Http, root: str, depth: int, cap: int) -> list[str]:
    """Breadth-first walk of the category graph, cycle-safe and capped."""
    seen = {root}
    order = [root]
    frontier = [root]
    for _ in range(depth):
        nxt: list[str] = []
        for parent in frontier:
            if len(order) >= cap:
                break
            cont: dict[str, Any] = {}
            while True:
                payload = http.api(
                    {
                        "action": "query",
                        "list": "categorymembers",
                        "cmtitle": parent,
                        "cmtype": "subcat",
                        "cmlimit": "500",
                        **cont,
                    }
                )
                for member in payload.get("query", {}).get("categorymembers", []):
                    title = member["title"]
                    if title not in seen:
                        seen.add(title)
                        order.append(title)
                        nxt.append(title)
                if "continue" not in payload or len(order) >= cap:
                    break
                cont = payload["continue"]
        frontier = nxt
        if not frontier:
            break
    return order[:cap]


def enumerate_entry(
    http: Http,
    entry: dict[str, Any],
    batch: int,
    subcat_cap: int,
    min_side: int,
    shuffle: bool = False,
    files_per_category: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield raw imageinfo page records for one plan entry.

    ``shuffle`` and ``files_per_category`` exist because categorymembers returns
    files in sort-key order, which is close enough to alphabetical that draining
    one category start-to-finish hands you a thousand filenames beginning with
    "A". Shuffling the subcategory list and taking a few files from each spreads
    a small sample over the whole subtree instead.
    """
    mode = entry.get("mode", "category")
    if mode == "category":
        cats = subcategories(http, entry["category"], int(entry.get("depth", 2)), subcat_cap)
        if shuffle:
            random.shuffle(cats)
        print(f"[plan] {entry['name']}: {len(cats)} categories", file=sys.stderr)
        for cat in cats:
            cont: dict[str, Any] = {}
            taken = 0
            while True:
                payload = http.api(
                    {
                        "action": "query",
                        "generator": "categorymembers",
                        "gcmtitle": cat,
                        "gcmtype": "file",
                        "gcmlimit": str(batch),
                        **_imageinfo_params(),
                        **cont,
                    }
                )
                pages = payload.get("query", {}).get("pages", [])
                if shuffle:
                    random.shuffle(pages)
                for page in pages:
                    yield page
                    taken += 1
                    if files_per_category and taken >= files_per_category:
                        break
                if files_per_category and taken >= files_per_category:
                    break
                if "continue" not in payload:
                    break
                cont = payload["continue"]
    elif mode == "search":
        # Search is the one mode that can filter resolution and format
        # server-side, which saves an enormous amount of metadata traffic.
        # Only augment if the entry has not set its own file constraints.
        query = entry["search"]
        if "filew:" not in query and "filewidth:" not in query:
            query = (
                f"{query} filetype:bitmap -filemime:image/gif filew:>{min_side} fileh:>{min_side}"
            )
        cont = {}
        while True:
            try:
                payload = http.api(
                    {
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": query,
                        "gsrnamespace": "6",
                        "gsrlimit": str(batch),
                        **_imageinfo_params(),
                        **cont,
                    }
                )
            except ApiError as exc:
                # CirrusSearch refuses deep pagination past ~10k results. That
                # is the ceiling per search entry; split the query if you need
                # more (e.g. add a filesize: or date range).
                if "offset" in str(exc).lower() or exc.code == "invalidparammix":
                    print(
                        f"[plan] {entry['name']}: pagination limit reached",
                        file=sys.stderr,
                    )
                    return
                raise
            yield from payload.get("query", {}).get("pages", [])
            if "continue" not in payload:
                return
            cont = payload["continue"]
    else:
        raise SystemExit(f"unknown mode {mode!r} in entry {entry['name']!r}")


# --------------------------------------------------------------------------
# candidate screening
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """Everything the screening step needs, so it stays a pure function."""

    min_side: int
    max_aspect: float
    fetch: str  # "original" | "thumb"
    max_bytes: int
    allow_junk_names: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Policy:
        return cls(
            min_side=args.min_side,
            max_aspect=args.max_aspect,
            fetch=args.fetch,
            max_bytes=args.max_bytes,
            allow_junk_names=args.allow_junk_names,
        )


@dataclass
class Candidate:
    name: str
    sha1: str
    url: str
    thumb: str
    width: int
    height: int
    mime: str
    page_url: str
    attribution: dict[str, str]
    entry: str


def screen(page: dict[str, Any], entry: str, policy: Policy) -> tuple[Candidate | None, str]:
    name = page.get("title", "")
    info = (page.get("imageinfo") or [{}])[0]
    if not info:
        return None, "no-imageinfo"
    if not policy.allow_junk_names and looks_like_junk(name):
        return None, "junk-name"

    mime = info.get("mime", "")
    if mime not in ALLOWED_MIME:
        return None, f"mime:{mime}"

    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if min(width, height) < policy.min_side:
        return None, f"small:{width}x{height}"
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > policy.max_aspect:
        # Panoramas, filmstrips and banners are poor detection data. In
        # `--fetch thumb` mode this also matters mechanically: past ~2.56 no
        # permitted thumbnail width preserves min_side.
        return None, f"aspect:{aspect:.1f}"

    ok, verdict, attribution = classify_license(info.get("extmetadata") or {})
    if not ok:
        return None, verdict

    url = (info.get("url") or "").split("?", 1)[0]
    if not url:
        return None, "no-url"
    thumb, _ = pick_url(url, width, height, int(info.get("size") or 0), policy)

    return (
        Candidate(
            name=name,
            sha1=info.get("sha1", ""),
            url=url,
            thumb=thumb,
            width=width,
            height=height,
            mime=mime,
            page_url=(info.get("descriptionurl") or "").split("?", 1)[0],
            attribution=attribution,
            entry=entry,
        ),
        "ok",
    )


# --------------------------------------------------------------------------
# crawl
# --------------------------------------------------------------------------


def run(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    state = State(out / "state.db")
    pause = GlobalPause()
    http = Http(args.user_agent, pause, token=args.api_token)
    policy = Policy.from_args(args)
    budget = ByteBudget(args.rate_mbps)
    manifest = (out / "manifest.jsonl").open("a", encoding="utf-8")
    manifest_lock = threading.Lock()

    work: queue.Queue[Candidate | None] = queue.Queue(maxsize=8 * args.concurrency)
    counters = {"accepted": 0, "rejected": 0, "failed": 0, "bytes": 0}
    counters_lock = threading.Lock()
    # `inflight` is what stops an entry overshooting its target: the producer
    # runs well ahead of the two download threads, so counting only completed
    # downloads would let it queue hundreds more than asked for.
    entry_accepted = {"n": 0, "inflight": 0}
    inflight_sha1: set[str] = set()
    stop = threading.Event()

    def download_worker() -> None:
        while not stop.is_set():
            item = work.get()
            try:
                if item is None:
                    return
                try:
                    _handle(item)
                finally:
                    with counters_lock:
                        entry_accepted["inflight"] -= 1
                        inflight_sha1.discard(item.sha1)
            finally:
                work.task_done()

    def _handle(cand: Candidate) -> None:
        try:
            data = http.fetch(cand.thumb, budget, args.max_bytes)
        except TooBig:
            state.record(name=cand.name, status="skip", reason="too-big", entry=cand.entry)
            return
        except urllib.error.HTTPError as exc:
            # A missing thumbnail (404) usually means the render failed; the
            # original is still there, but it is not worth the bytes.
            state.record(name=cand.name, status="fail", reason=f"http:{exc.code}", entry=cand.entry)
            with counters_lock:
                counters["failed"] += 1
            return
        except Exception as exc:
            state.record(
                name=cand.name,
                status="fail",
                reason=f"net:{type(exc).__name__}",
                entry=cand.entry,
            )
            with counters_lock:
                counters["failed"] += 1
            return

        ok, verdict, size = verify(data, args.min_side)
        if not ok:
            state.record(name=cand.name, status="skip", reason=verdict, entry=cand.entry)
            with counters_lock:
                counters["rejected"] += 1
            return

        downloaded = len(data)  # count what crossed the wire, before any resize
        got_w, got_h = size if size else (cand.width, cand.height)
        mime = cand.mime
        if args.resize_to and min(got_w, got_h) > args.resize_to:
            try:
                data, (got_w, got_h) = downscale(data, args.resize_to, args.jpeg_quality)
                mime = "image/jpeg"
            except Exception as exc:
                print(f"[warn] resize failed for {cand.name}: {exc}", file=sys.stderr)

        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
        sha = cand.sha1 or f"nohash{abs(hash(cand.name)):016x}"
        rel = Path("images") / sha[:2] / sha[2:4] / f"{sha}{suffix}"
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)

        record = {
            "file": cand.name,
            "path": str(rel),
            "sha1": cand.sha1,
            "source": cand.page_url,
            "url": cand.thumb,
            "orig_width": cand.width,
            "orig_height": cand.height,
            "width": got_w,
            "height": got_h,
            "bytes": len(data),
            "mime": mime,
            "entry": cand.entry,
            **cand.attribution,
        }
        with manifest_lock:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
        state.record(
            name=cand.name,
            sha1=cand.sha1,
            status="done",
            entry=cand.entry,
            width=got_w,
            height=got_h,
            path=str(rel),
        )
        state.bump_entry(cand.entry, accepted=1)
        with counters_lock:
            counters["accepted"] += 1
            counters["bytes"] += downloaded
            entry_accepted["n"] += 1

    workers = [
        threading.Thread(target=download_worker, daemon=True) for _ in range(args.concurrency)
    ]
    for worker in workers:
        worker.start()

    started = time.time()
    try:
        for entry in plan:
            name = entry["name"]
            already, done = state.entry_progress(name)
            target = int(entry.get("target", args.target))
            if done or already >= target:
                print(f"[skip] {name}: already {already}/{target}", file=sys.stderr)
                continue
            entry_accepted["n"] = already
            print(f"[entry] {name} -> {target} images", file=sys.stderr)

            exhausted = True
            try:
                pages = enumerate_entry(
                    http,
                    entry,
                    args.batch,
                    args.subcat_cap,
                    args.min_side,
                    args.shuffle,
                    args.files_per_category,
                )
                for page in pages:
                    with counters_lock:
                        reached = entry_accepted["n"] + entry_accepted["inflight"] >= target
                    if reached:
                        exhausted = False
                        break
                    title = page.get("title", "")
                    if not title or state.seen(title):
                        continue
                    cand, reason = screen(page, name, policy)
                    if cand is None:
                        state.record(name=title, status="skip", reason=reason, entry=name)
                        with counters_lock:
                            counters["rejected"] += 1
                        continue
                    # `inflight_sha1` closes a race the DB cannot: a duplicate
                    # queued while its twin is still downloading is not yet
                    # 'done', so state.sha1_taken() would wave it through.
                    with counters_lock:
                        dup = cand.sha1 in inflight_sha1
                    if cand.sha1 and (dup or state.sha1_taken(cand.sha1)):
                        state.record(name=title, status="skip", reason="dup", entry=name)
                        continue
                    if args.dry_run:
                        print(f"  would fetch {cand.thumb}")
                        with counters_lock:
                            entry_accepted["n"] += 1
                        continue
                    with counters_lock:
                        entry_accepted["inflight"] += 1
                        inflight_sha1.add(cand.sha1)
                    work.put(cand)
                    _progress(counters, counters_lock, started)
            except Exception as exc:
                # A crawl measured in days will hit these. Losing the rest of
                # one entry is survivable; losing the run is not.
                print(f"[warn] {name} aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
                exhausted = False

            work.join()
            with counters_lock:
                final = entry_accepted["n"]
            state.bump_entry(name, done=exhausted)  # count is bumped per image
            print(
                f"[entry] {name}: {final} accepted{' (exhausted)' if exhausted else ''}",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print("\n[stop] interrupted — state saved, re-run to resume", file=sys.stderr)
    finally:
        stop.set()
        for _ in workers:
            work.put(None)
        for worker in workers:
            worker.join(timeout=5.0)
        manifest.close()

    elapsed = max(1e-6, time.time() - started)
    print(
        f"\naccepted={counters['accepted']} rejected={counters['rejected']} "
        f"failed={counters['failed']} "
        f"{counters['bytes'] / 1e9:.2f} GB in {elapsed / 60:.1f} min "
        f"({counters['accepted'] / elapsed:.1f} img/s)",
        file=sys.stderr,
    )
    print(f"totals in state.db: {state.totals()}", file=sys.stderr)


_last_print = [0.0]


def _progress(counters: dict[str, int], lock: threading.Lock, started: float) -> None:
    now = time.time()
    if now - _last_print[0] < 5.0:
        return
    _last_print[0] = now
    with lock:
        accepted, rejected, nbytes = (
            counters["accepted"],
            counters["rejected"],
            counters["bytes"],
        )
    elapsed = max(1e-6, now - started)
    print(
        f"  {accepted} kept / {rejected} dropped | "
        f"{nbytes / 1e9:.2f} GB | {accepted / elapsed:.1f} img/s",
        file=sys.stderr,
        flush=True,
    )


# --------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------


def estimate(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    """Sample each plan entry and report the pass rate before committing."""
    http = Http(args.user_agent, GlobalPause(), token=args.api_token)
    policy = Policy.from_args(args)
    sample = args.sample
    print(
        f"{'entry':<28} {'sampled':>8} {'kept':>6} {'rate':>6}  top rejection",
        file=sys.stderr,
    )
    print("-" * 78, file=sys.stderr)
    grand_seen = grand_kept = 0
    for entry in plan:
        seen = kept = 0
        reasons: dict[str, int] = {}
        try:
            for page in enumerate_entry(
                http,
                entry,
                args.batch,
                args.subcat_cap,
                args.min_side,
                args.shuffle,
                args.files_per_category,
            ):
                seen += 1
                cand, reason = screen(page, entry["name"], policy)
                if cand is not None:
                    kept += 1
                else:
                    key = reason.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
                if seen >= sample:
                    break
        except Exception as exc:
            print(f"{entry['name']:<28} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        grand_seen += seen
        grand_kept += kept
        top = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "-"
        rate = f"{100 * kept / seen:.1f}%" if seen else "-"
        print(f"{entry['name']:<28} {seen:>8} {kept:>6} {rate:>6}  {top}", file=sys.stderr)
    overall = f"{100 * grand_kept / grand_seen:.1f}%" if grand_seen else "-"
    print("-" * 78, file=sys.stderr)
    print(f"{'OVERALL':<28} {grand_seen:>8} {grand_kept:>6} {overall:>6}", file=sys.stderr)
    print(
        "\nRates are from the first N members of each category, which skew "
        "alphabetically — treat as a rough yield signal, not a census.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

_UA_CONTACT = re.compile(r"https?://|[^\s@]+@[^\s@]+\.[^\s@]+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="commons_pool", help="output directory")
    parser.add_argument("--plan", help="JSON plan file (default: built-in plan)")
    parser.add_argument("--dump-plan", metavar="PATH", help="write the default plan and exit")
    parser.add_argument(
        "--user-agent",
        help="required; must name the tool and carry a URL or e-mail contact",
    )
    parser.add_argument("--api-token", help="OAuth2 bearer token, raises API rate limits")
    parser.add_argument("--min-side", type=int, default=1500, help="minimum shortest side, px")
    parser.add_argument(
        "--fetch",
        choices=("original", "thumb"),
        default="original",
        help="original: full resolution (default). thumb: smallest permitted "
        "thumbnail that still clears --min-side, ~4x cheaper in bytes",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=2.5,
        help="reject images longer than this ratio (panoramas, banners)",
    )
    parser.add_argument("--target", type=int, default=20000, help="default per-entry cap")
    parser.add_argument(
        "--batch", type=int, default=50, help="API page size (500 needs apihighlimits)"
    )
    parser.add_argument("--subcat-cap", type=int, default=400, help="max subcats per entry")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="randomise subcategory and page order; without it a small sample is "
        "drawn alphabetically from the first subcategory of each entry",
    )
    parser.add_argument(
        "--files-per-category",
        type=int,
        default=0,
        metavar="N",
        help="look at at most N files per subcategory before moving on (0 = all). "
        "Use with --shuffle to spread a small sample across a whole subtree",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="download connections (policy max: 2)"
    )
    parser.add_argument(
        "--rate-mbps", type=float, default=20.0, help="download ceiling (policy max: 25)"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=25_000_000,
        help="above this, step down to the largest qualifying thumbnail",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=0,
        metavar="PX",
        help="downscale so the shortest side is PX (0 = keep as downloaded). "
        "Recommended: same value as --min-side; needs Pillow",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92, help="for --resize-to")
    parser.add_argument(
        "--allow-junk-names",
        action="store_true",
        help="keep maps/scans/heraldry that the filename filter would drop",
    )
    parser.add_argument("--estimate", action="store_true", help="sample yield and exit")
    parser.add_argument("--sample", type=int, default=200, help="files sampled per entry")
    parser.add_argument("--dry-run", action="store_true", help="screen but do not download")
    args = parser.parse_args(argv)

    if args.dump_plan:
        Path(args.dump_plan).write_text(
            json.dumps(default_plan(args.target), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.dump_plan}")
        return 0

    if not args.user_agent or not _UA_CONTACT.search(args.user_agent):
        parser.error(
            "--user-agent is required and must include a contact URL or e-mail, e.g.\n"
            '  --user-agent "mayaku-harvest/1.0 '
            '(https://github.com/datamarkin/mayaku; you@example.com)"\n'
            "Wikimedia blocks generic agents: https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy"
        )
    if args.resize_to:
        try:
            import PIL  # noqa: F401
        except ImportError:
            parser.error("--resize-to needs Pillow: pip install pillow")
    if args.concurrency > 2:
        parser.error("robot policy caps upload.wikimedia.org at 2 concurrent connections")
    if args.rate_mbps > 25:
        parser.error("robot policy caps download bandwidth at 25 Mbps")

    plan = (
        json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if args.plan
        else default_plan(args.target)
    )

    if args.estimate:
        estimate(args, plan)
    else:
        run(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
