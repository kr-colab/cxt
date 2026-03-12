#!/usr/bin/env python3
"""
Verify bibliography entries against CrossRef / DOI metadata.

For each entry in the .bib file that is actually cited (appears in the .bbl),
this script:
  1. Looks up the DOI via the CrossRef API
  2. For entries without a DOI, searches CrossRef by title
  3. Compares returned metadata (title, year, volume, first-page) against
     the .bib values
  4. Produces a colour-coded report showing MATCH / MISMATCH / UNVERIFIED

Usage:
    python verify_doi.py                   # uses defaults in cxt_paper/
    python verify_doi.py path/to/cxt.bib path/to/Research_report.bbl

Requires: requests  (pip install requests)
"""

import re
import sys
import os
import json
import time
import unicodedata
import urllib.parse

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' package required.  Install with:  pip install requests")

CROSSREF_BASE = "https://api.crossref.org"
HEADERS = {
    "User-Agent": "PNASBibVerifier/1.0 (mailto:kkor@uoregon.edu)",
    "Accept": "application/json",
}
RATE_LIMIT_SLEEP = 0.15  # polite delay between API calls


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(text):
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[{}\\'\"\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def first_page(pages_str):
    """Extract the first page number from a pages string like '235--248'."""
    m = re.match(r"(\d+)", pages_str.strip())
    return m.group(1) if m else ""


def strip_bibtex_braces(val):
    """Remove BibTeX braces and common LaTeX accents."""
    val = re.sub(r"\\['`\"^~=.uvHtcdb]\{?(\w)\}?", r"\1", val)
    val = val.replace("{", "").replace("}", "")
    return val.strip()


# ── .bib parser ──────────────────────────────────────────────────────────────

def parse_bib(path):
    """Return dict of cite_key -> {field: value} from a .bib file."""
    with open(path) as f:
        content = f.read()

    entries = {}
    entry_re = re.compile(
        r"@(\w+)\s*\{([^,]+),(.+?)(?=\n@|\Z)", re.DOTALL
    )
    for m in entry_re.finditer(content):
        etype = m.group(1).lower()
        key = m.group(2).strip()
        body = m.group(3)
        fields = {"_type": etype}

        field_re = re.compile(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|(\d+))",
            re.DOTALL,
        )
        for fm in field_re.finditer(body):
            fname = fm.group(1).lower()
            fval = fm.group(2) if fm.group(2) is not None else fm.group(3)
            fval = re.sub(r"\s+", " ", fval).strip()
            fields[fname] = fval

        entries[key] = fields
    return entries


def cited_keys(bbl_path):
    """Return ordered list of cite keys from a .bbl file."""
    with open(bbl_path) as f:
        content = f.read()
    return re.findall(r"\\bibitem\{([^}]+)\}", content)


# ── CrossRef lookups ─────────────────────────────────────────────────────────

def lookup_doi(doi):
    """Fetch metadata from CrossRef for a given DOI. Returns dict or None."""
    url = f"{CROSSREF_BASE}/works/{urllib.parse.quote(doi, safe='')}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("message", {})
        return None
    except Exception:
        return None


def search_title(title):
    """Search CrossRef by title. Returns best-match metadata dict or None."""
    clean = re.sub(r"[{}\\]", "", title)[:200]
    params = {"query.title": clean, "rows": 1}
    try:
        resp = requests.get(
            f"{CROSSREF_BASE}/works", params=params, headers=HEADERS, timeout=20
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            if items:
                return items[0]
        return None
    except Exception:
        return None


# ── comparison logic ─────────────────────────────────────────────────────────

def compare(bib_entry, cr_meta):
    """Compare .bib fields against CrossRef metadata. Returns list of (field, status, bib_val, cr_val)."""
    results = []

    # title
    cr_titles = cr_meta.get("title", [])
    cr_title = cr_titles[0] if cr_titles else ""
    bib_title = strip_bibtex_braces(bib_entry.get("title", ""))
    t_match = normalize(bib_title)[:60] in normalize(cr_title) or normalize(cr_title)[:60] in normalize(bib_title)
    results.append(("title", "ok" if t_match else "MISMATCH", bib_title[:80], cr_title[:80]))

    # year
    cr_year = ""
    for date_field in ("published-print", "published-online", "published", "created"):
        dp = cr_meta.get(date_field, {}).get("date-parts", [[]])
        if dp and dp[0]:
            cr_year = str(dp[0][0])
            break
    bib_year = bib_entry.get("year", "")
    results.append(("year", "ok" if bib_year == cr_year else "MISMATCH", bib_year, cr_year))

    # volume
    cr_vol = cr_meta.get("volume", "")
    bib_vol = bib_entry.get("volume", "")
    if bib_vol or cr_vol:
        results.append(("volume", "ok" if bib_vol == cr_vol else "MISMATCH", bib_vol, cr_vol))

    # first page
    cr_page = cr_meta.get("page", "")
    cr_fp = first_page(cr_page) if cr_page else ""
    cr_article = cr_meta.get("article-number", "")
    bib_pages = bib_entry.get("pages", "")
    bib_fp = first_page(bib_pages)
    bib_eid = bib_entry.get("eid", "")
    if bib_fp or cr_fp or cr_article:
        page_ok = False
        if bib_fp and cr_fp and bib_fp == cr_fp:
            page_ok = True
        if bib_pages and cr_article and (cr_article in bib_pages or bib_pages in cr_article):
            page_ok = True
        if bib_eid and cr_article and (cr_article in bib_eid or bib_eid in cr_article):
            page_ok = True
        if not cr_fp and not cr_article:
            page_ok = True  # CrossRef missing pages is not our problem
        cr_display = cr_page or cr_article or ""
        bib_display = bib_pages or bib_eid or ""
        results.append(("pages", "ok" if page_ok else "MISMATCH", bib_display, cr_display))

    # DOI match (sanity check that we got the right record)
    cr_doi = cr_meta.get("DOI", "").lower()
    bib_doi = bib_entry.get("doi", "").lower()
    if bib_doi and cr_doi:
        # strip common prefixes
        bib_doi_clean = re.sub(r"^https?://doi\.org/", "", bib_doi)
        cr_doi_clean = re.sub(r"^https?://doi\.org/", "", cr_doi)
        results.append(("doi", "ok" if bib_doi_clean == cr_doi_clean else "MISMATCH", bib_doi, cr_doi))

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bib_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "cxt.bib")
    bbl_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, "Research_report.bbl")

    if not os.path.exists(bib_path):
        sys.exit(f"ERROR: {bib_path} not found")
    if not os.path.exists(bbl_path):
        sys.exit(f"ERROR: {bbl_path} not found (compile the document first)")

    bib = parse_bib(bib_path)
    keys = cited_keys(bbl_path)

    print("=" * 80)
    print("DOI / CROSSREF VERIFICATION REPORT")
    print("=" * 80)
    print(f"Bib entries: {len(bib)} | Cited: {len(keys)}")
    print()

    stats = {"verified": 0, "mismatch": 0, "not_found": 0, "no_doi_verified": 0, "no_doi_not_found": 0}
    all_results = []

    for idx, key in enumerate(keys, 1):
        entry = bib.get(key)
        if not entry:
            print(f"[{idx:2d}] {key}  -- NOT IN .bib FILE")
            stats["not_found"] += 1
            all_results.append((idx, key, "NOT_IN_BIB", []))
            continue

        doi = entry.get("doi", "")
        title = entry.get("title", "")
        method = ""
        cr = None

        # try DOI first
        if doi:
            clean_doi = re.sub(r"^https?://doi\.org/", "", doi)
            cr = lookup_doi(clean_doi)
            method = f"DOI: {clean_doi}"
            time.sleep(RATE_LIMIT_SLEEP)

        # fallback to title search
        if cr is None and title:
            cr = search_title(title)
            method = "title-search"
            time.sleep(RATE_LIMIT_SLEEP)

        if cr is None:
            print(f"[{idx:2d}] {key}")
            print(f"      method: {method or 'none'}")
            print(f"      result: COULD NOT VERIFY (API returned nothing)")
            print()
            if doi:
                stats["not_found"] += 1
            else:
                stats["no_doi_not_found"] += 1
            all_results.append((idx, key, "NOT_FOUND", []))
            continue

        checks = compare(entry, cr)
        has_mismatch = any(c[1] == "MISMATCH" for c in checks)

        status = "MISMATCH" if has_mismatch else "VERIFIED"
        if has_mismatch:
            stats["mismatch"] += 1
        elif doi:
            stats["verified"] += 1
        else:
            stats["no_doi_verified"] += 1

        marker = "***" if has_mismatch else "   "
        print(f"[{idx:2d}] {key}  {marker}")
        print(f"      method: {method}")
        for field, fstatus, bval, cval in checks:
            tag = "OK" if fstatus == "ok" else "MISMATCH"
            if tag == "MISMATCH":
                print(f"      {field:8s}: {tag}  bib='{bval}'  crossref='{cval}'")
            else:
                print(f"      {field:8s}: {tag}")
        print()
        all_results.append((idx, key, status, checks))

    # summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Verified via DOI:           {stats['verified']}")
    print(f"  Verified via title search:  {stats['no_doi_verified']}")
    print(f"  Mismatches found:           {stats['mismatch']}")
    print(f"  Could not verify:           {stats['not_found'] + stats['no_doi_not_found']}")
    print()

    if stats["mismatch"]:
        print("ENTRIES WITH MISMATCHES:")
        for idx, key, status, checks in all_results:
            if status == "MISMATCH":
                print(f"  [{idx:2d}] {key}")
                for field, fstatus, bval, cval in checks:
                    if fstatus == "MISMATCH":
                        print(f"        {field}: bib='{bval}' vs crossref='{cval}'")
        print()

    unverified = [(idx, key) for idx, key, status, _ in all_results if status == "NOT_FOUND"]
    if unverified:
        print("UNVERIFIABLE ENTRIES (no DOI and title search failed):")
        for idx, key in unverified:
            print(f"  [{idx:2d}] {key}")
        print()

    print("NOTE: 'MISMATCH' in year can occur when CrossRef stores the")
    print("online-first date rather than the print date. Small page/volume")
    print("differences may reflect corrections or eid vs page numbering.")
    print("Review mismatches manually to confirm they are not errors.")

    # write JSON for machine consumption
    json_path = os.path.join(script_dir, "doi_verification.json")
    json_out = []
    for idx, key, status, checks in all_results:
        entry_out = {"ref_num": idx, "key": key, "status": status, "checks": []}
        for field, fstatus, bval, cval in checks:
            entry_out["checks"].append({
                "field": field, "status": fstatus,
                "bib_value": bval, "crossref_value": cval
            })
        json_out.append(entry_out)
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nDetailed results written to: {json_path}")

    return 1 if stats["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
