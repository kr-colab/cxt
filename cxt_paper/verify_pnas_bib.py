#!/usr/bin/env python3
"""
Verify PNAS bibliography compliance.

Parses the compiled .bbl file, strips LaTeX formatting, and checks each
reference for common PNAS style issues.  Produces a numbered plain-text
reference list plus a summary of flagged problems.

Usage:
    python verify_pnas_bib.py [path/to/file.bbl]

If no path is given, defaults to Research_report.bbl in the same directory.
"""

import re
import sys
import os

KNOWN_FULL_JOURNALS = [
    "Molecular Biology and Evolution",
    "Molecular biology and evolution",
    "Nature Genetics",
    "Nature genetics",
    "Theoretical Population Biology",
    "Journal of Computational Biology",
    "Philosophical Transactions",
    "Molecular Ecology Resources",
    "Molecular Ecology",
    "Genome Biology and Evolution",
    "PLoS computational biology",
    "PLoS Computational Biology",
    "PLOS Genetics",
    "PLoS Genetics",
    "Nature Biotechnology",
    "Nature Reviews Genetics",
    "Genome research",
    "Annual review of genetics",
    "The American Journal of Human Genetics",
    "Human genomics",
    "Insect molecular biology",
    "Proceedings of the National Academy",
    "Communications Biology",
]

PROPER_NOUNS = [
    "Bayesian", "Markovian", "Drosophila", "Anopheles", "African",
    "European", "Aboriginal", "Australia", "DNA", "GABA", "MHC",
    "ABC", "Paleolithic", "Neolithic", "Rdl", "AI",
]


def strip_latex(text):
    """Remove LaTeX commands to produce readable plain text."""
    text = text.replace("\n", " ")
    text = re.sub(r"\\newblock\s*", "", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\{\\em\\protect\\JournalTitle\{([^}]*)\}\}", r"\1", text)
    text = re.sub(r"\{\\em\s+([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\doi\{([^}]*)\}", r"doi:\1", text)
    text = re.sub(r"\\'\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"et~al\.", "et al.", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def parse_bbl(path):
    """Return list of (cite_key, raw_latex) pairs from a .bbl file."""
    with open(path) as f:
        content = f.read()

    entries = []
    parts = re.split(r"\\bibitem\{([^}]+)\}", content)
    for i in range(1, len(parts), 2):
        key = parts[i]
        raw = parts[i + 1].split("\\bibitem{")[0].strip()
        raw = raw.split("\\end{thebibliography}")[0].strip()
        entries.append((key, raw))
    return entries


def check_initials(text):
    """Flag author initials that lack periods (e.g. 'J ' instead of 'J. ')."""
    author_part = text.split(",")[0] if "," in text else text[:60]
    issues = []
    matches = re.findall(r"(?<![.\w])([A-Z])(?=[A-Z]|\s[A-Z][a-z])", author_part)
    if matches:
        issues.append(f"possible missing period after initial(s): {''.join(matches)}")
    return issues


def check_journal(text):
    """Flag unabbreviated journal names."""
    issues = []
    for name in KNOWN_FULL_JOURNALS:
        if name in text:
            issues.append(f"unabbreviated journal: \"{name}\"")
            break
    return issues


def check_proper_nouns(text):
    """Flag proper nouns that appear lowercase in the title portion."""
    issues = []
    title_match = re.match(r"^[^.]+\.", text)
    if not title_match:
        return issues
    title_text = title_match.group(0)
    for noun in PROPER_NOUNS:
        lower = noun.lower()
        if lower in title_text.lower():
            if noun not in title_text:
                issues.append(f"'{lower}' should be '{noun}' in title")
    return issues


def check_garbled(text):
    """Flag garbled author names (double commas, stray periods at start)."""
    issues = []
    if ", , " in text:
        issues.append("garbled author (double comma)")
    if re.match(r"^\.\s", text):
        issues.append("garbled author (leading period)")
    return issues


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_bbl = os.path.join(script_dir, "Research_report.bbl")
    bbl_path = sys.argv[1] if len(sys.argv) > 1 else default_bbl

    if not os.path.exists(bbl_path):
        print(f"ERROR: {bbl_path} not found. Compile the document first.")
        sys.exit(1)

    entries = parse_bbl(bbl_path)

    print("=" * 78)
    print("PNAS BIBLIOGRAPHY VERIFICATION REPORT")
    print("=" * 78)
    print()

    all_issues = []
    for idx, (key, raw) in enumerate(entries, 1):
        clean = strip_latex(raw)
        issues = []
        issues.extend(check_initials(clean))
        issues.extend(check_journal(clean))
        issues.extend(check_proper_nouns(clean))
        issues.extend(check_garbled(clean))

        marker = "  *** " if issues else "      "
        print(f"[{idx:2d}] {clean}")
        if issues:
            for iss in issues:
                print(f"      >> {iss}")
            print()
        all_issues.append((idx, key, issues))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    problem_count = sum(1 for _, _, iss in all_issues if iss)
    total = len(entries)
    print(f"Total references: {total}")
    print(f"References with issues: {problem_count}")
    print(f"Clean references: {total - problem_count}")

    if problem_count:
        print()
        print("Issues by reference:")
        for idx, key, issues in all_issues:
            if issues:
                print(f"  [{idx:2d}] {key}")
                for iss in issues:
                    print(f"        - {iss}")
    else:
        print()
        print("All references pass automated checks.")

    print()
    print("NOTE: This script checks for common issues but cannot catch")
    print("everything. Please visually scan the numbered list above for")
    print("any remaining problems (unusual formatting, missing fields, etc.).")

    return 1 if problem_count else 0


if __name__ == "__main__":
    sys.exit(main())
