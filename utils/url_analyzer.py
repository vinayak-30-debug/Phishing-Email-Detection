"""
utils/url_analyzer.py
---------------------
Extract URLs from email text and analyse them for phishing-related risk signals.

Each URL is scored on multiple heuristics:
  • Shortened link (bit.ly, tinyurl, etc.)
  • IP-address host (http://192.168.1.1/…)
  • Look-alike / typosquat domain (paypa1.com, g00gle.com)
  • Excessive subdomains (login.secure.account.example.com)
  • Suspicious TLD (.xyz, .tk, .buzz, …)
  • '@' in URL (credential-embedding trick)
  • Very long URL (obfuscation)
  • Hex/encoded characters (%xx sequences)
  • Mismatched or absent HTTPS

The module exposes two public helpers consumed by app.py:

    extract_urls(text) → list[str]
    analyse_urls(urls)  → list[dict]   (one dict per URL with flags + score)
"""

from __future__ import annotations

import re
import math
from urllib.parse import urlparse, unquote


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Well-known URL shorteners
_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "rb.gy", "shorturl.at",
    "tiny.cc", "lnkd.in", "soo.gd", "s.id", "bl.ink", "clck.ru",
    "v.gd", "qr.ae", "adf.ly", "shorte.st", "bc.vc",
}

# Suspicious / cheap TLDs commonly abused in phishing
_SUSPICIOUS_TLDS: set[str] = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".buzz", ".top",
    ".club", ".work", ".click", ".link", ".info", ".online",
    ".site", ".icu", ".cam", ".rest", ".monster", ".fun",
    ".space", ".store", ".loan", ".win", ".racing", ".download",
    ".stream", ".accountant", ".date", ".faith", ".review",
}

# Domains that are commonly spoofed (used for look-alike detection)
_TARGET_BRANDS: list[str] = [
    "google", "facebook", "apple", "microsoft", "amazon", "paypal",
    "netflix", "instagram", "twitter", "linkedin", "dropbox", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "americanexpress", "usps", "fedex", "dhl", "ups",
]

# Simple leet-speak substitutions for look-alike detection
_LEET_MAP: dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s", "!": "i",
}

# ──────────────────────────────────────────────────────────────────────────────
# URL extraction
# ──────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"https?://[^\s<>\"'`\)\]\}]+",
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    """Return a de-duplicated, order-preserving list of HTTP(S) URLs in *text*."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_ip_host(hostname: str) -> bool:
    """True if *hostname* looks like an IPv4 address."""
    parts = hostname.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def _deleet(domain: str) -> str:
    """Reverse leet-speak in *domain* so 'g00gle' becomes 'google'."""
    return "".join(_LEET_MAP.get(ch, ch) for ch in domain)


def _is_lookalike(hostname: str) -> str | None:
    """
    Check whether *hostname* is a look-alike of a well-known brand.
    Returns the brand name if a match is found, otherwise None.
    """
    # Strip TLD to get the core domain label
    parts = hostname.lower().split(".")
    if len(parts) < 2:
        return None

    # Check the registrable domain label (second-to-last part for simple TLDs)
    domain_label = parts[-2]
    deleeted = _deleet(domain_label)

    for brand in _TARGET_BRANDS:
        # Exact match after de-leeting → look-alike
        if deleeted == brand and domain_label != brand:
            return brand
        # Substring tricks: "paypal-secure", "login-google", etc.
        if brand in domain_label and domain_label != brand:
            return brand
        # Edit-distance 1 (simple typosquat: "gooogle", "paypall")
        if _edit_distance_one(deleeted, brand):
            return brand

    return None


def _edit_distance_one(a: str, b: str) -> bool:
    """True if strings *a* and *b* have Levenshtein distance exactly 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(ca != cb for ca, cb in zip(a, b)) == 1
    # One is longer by exactly 1 char
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    diffs = 0
    si = li = 0
    while si < len(short) and li < len(long_):
        if short[si] != long_[li]:
            diffs += 1
            li += 1
        else:
            si += 1
            li += 1
    return diffs <= 1


def _count_hex_chars(url: str) -> int:
    """Count percent-encoded sequences (%xx) in *url*."""
    return len(re.findall(r"%[0-9A-Fa-f]{2}", url))


# ──────────────────────────────────────────────────────────────────────────────
# Public analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_urls(urls: list[str]) -> list[dict]:
    """
    Analyse a list of URLs and return a list of result dicts.

    Each dict contains:
        url          – the original URL string
        hostname     – extracted hostname
        flags        – list[dict] of triggered signals, each with:
                           label  (str)  – human-readable description
                           weight (int)  – risk points added (1-3)
        risk_score   – 0-100 normalised risk score
        risk_level   – "low" | "medium" | "high" | "critical"
    """
    results: list[dict] = []

    for url in urls:
        flags: list[dict] = []

        try:
            parsed = urlparse(url)
        except Exception:
            flags.append({"label": "Unparseable URL", "weight": 3})
            results.append({
                "url": url,
                "hostname": "—",
                "flags": flags,
                "risk_score": 100,
                "risk_level": "critical",
            })
            continue

        hostname = (parsed.hostname or "").lower()

        # 1. No HTTPS
        if parsed.scheme != "https":
            flags.append({"label": "Not using HTTPS", "weight": 1})

        # 2. IP-address host
        if _is_ip_host(hostname):
            flags.append({"label": "IP address as host", "weight": 3})

        # 3. Shortened URL
        if hostname in _SHORTENERS:
            flags.append({"label": "URL shortener", "weight": 2})

        # 4. Look-alike domain
        brand = _is_lookalike(hostname)
        if brand:
            flags.append({"label": f"Look-alike of '{brand}'", "weight": 3})

        # 5. Suspicious TLD
        for tld in _SUSPICIOUS_TLDS:
            if hostname.endswith(tld):
                flags.append({"label": f"Suspicious TLD ({tld})", "weight": 2})
                break

        # 6. Excessive subdomains (≥ 4 labels)
        labels = hostname.split(".")
        if len(labels) >= 4:
            flags.append({"label": f"Excessive subdomains ({len(labels)} levels)", "weight": 2})

        # 7. '@' in URL (credential-embedding)
        if "@" in url:
            flags.append({"label": "'@' in URL (credential trick)", "weight": 3})

        # 8. Very long URL (> 75 chars)
        if len(url) > 75:
            flags.append({"label": f"Very long URL ({len(url)} chars)", "weight": 1})

        # 9. Percent-encoded characters
        hex_count = _count_hex_chars(url)
        if hex_count >= 3:
            flags.append({"label": f"Heavy encoding ({hex_count} encoded chars)", "weight": 2})

        # 10. Suspicious path keywords
        path_lower = (parsed.path + "?" + (parsed.query or "")).lower()
        sus_keywords = ["login", "signin", "verify", "update", "secure",
                        "account", "confirm", "password", "credential", "wallet"]
        found_kw = [kw for kw in sus_keywords if kw in path_lower]
        if found_kw:
            flags.append({
                "label": f"Suspicious path keywords: {', '.join(found_kw)}",
                "weight": 2,
            })

        # ── Compute score ─────────────────────────────────────────────────
        raw = sum(f["weight"] for f in flags)
        # Normalise to 0-100 using a soft-cap sigmoid-style curve
        # so that even 2-3 flags push the score high, but it never exceeds 100.
        risk_score = int(100 * (1 - math.exp(-0.5 * raw)))

        if risk_score >= 80:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"

        results.append({
            "url": url,
            "hostname": hostname,
            "flags": flags,
            "risk_score": risk_score,
            "risk_level": risk_level,
        })

    return results


def overall_threat_level(analysis: list[dict]) -> str:
    """
    Given the output of ``analyse_urls``, return an aggregate threat string:
    'none', 'low', 'medium', 'high', or 'critical'.
    """
    if not analysis:
        return "none"
    worst = max(r["risk_score"] for r in analysis)
    if worst >= 80:
        return "critical"
    if worst >= 50:
        return "high"
    if worst >= 25:
        return "medium"
    return "low"
