#!/usr/bin/env python3
"""Single version for the whole dnalm-hub repo (all packages and services are released together).

    python3 scripts/version.py                  print the current version
    python3 scripts/version.py check [--tag T]  every file agrees (and matches tag T, e.g. v0.9.0)
    python3 scripts/version.py set 0.9.1        write a new version everywhere

`set` rewrites the `[project] version` of every pyproject.toml, the matching
entries in every uv.lock (no re-resolve, so other pins don't move), the ntv3
`__version__` and CITATION.cff. Stdlib only; run from anywhere. Usually via
`make version`, `make check-version` and `make bump VERSION=...`.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
CITATION = ROOT / "CITATION.cff"
INIT_FILES = [ROOT / "services/ntv3/src/ntv3_mcp/__init__.py"]


def projects() -> list[Path]:
    """Every project directory with a pyproject.toml (packages, services and the template)."""
    return sorted(p.parent for p in ROOT.glob("*/*/pyproject.toml") if p.parts[-3] != ".claude")


def project_name_version(pyproject: Path) -> tuple[str, str]:
    text = pyproject.read_text()
    section = text.split("[project]", 1)[1].split("\n[", 1)[0]
    name = re.search(r'^name\s*=\s*"([^"]+)"', section, re.MULTILINE)
    version = re.search(r'^version\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not name or not version:
        raise SystemExit(f"{pyproject}: no [project] name/version")
    return name.group(1), version.group(1)


def own_names() -> set[str]:
    return {project_name_version(p / "pyproject.toml")[0] for p in projects()}


def current_versions() -> dict[str, str]:
    """File (relative path, plus lock entry) -> version found there."""
    found: dict[str, str] = {}
    names = own_names()
    for proj in projects():
        rel = proj.relative_to(ROOT)
        found[f"{rel}/pyproject.toml"] = project_name_version(proj / "pyproject.toml")[1]
        lock = proj / "uv.lock"
        if lock.exists():
            for name, version in re.findall(
                r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"',
                lock.read_text(),
                re.MULTILINE,
            ):
                if name in names:
                    found[f"{rel}/uv.lock [{name}]"] = version
    for init in INIT_FILES:
        m = re.search(r'^__version__ = "([^"]+)"', init.read_text(), re.MULTILINE)
        if m:
            found[str(init.relative_to(ROOT))] = m.group(1)
    if CITATION.exists():
        m = re.search(r'^version: "?([^"\n]+)"?', CITATION.read_text(), re.MULTILINE)
        if m:
            found["CITATION.cff"] = m.group(1)
    return found


def hub_version() -> str:
    return project_name_version(ROOT / "packages/dnalm-common/pyproject.toml")[1]


def check(tag: str | None) -> int:
    expected = hub_version()
    if tag is not None and tag.removeprefix("v") != expected:
        print(f"tag {tag} does not match version {expected}", file=sys.stderr)
        return 1
    wrong = {f: v for f, v in current_versions().items() if v != expected}
    for f, v in sorted(wrong.items()):
        print(f"{f}: {v} (expected {expected})", file=sys.stderr)
    if wrong:
        print("run: make bump VERSION=" + expected, file=sys.stderr)
        return 1
    print(f"version {expected} consistent across {len(current_versions())} entries")
    return 0


def set_version(new: str) -> None:
    if not SEMVER.match(new):
        raise SystemExit(f"not a semantic version: {new!r} (expected e.g. 0.9.1)")
    names = own_names()
    for proj in projects():
        pyproject = proj / "pyproject.toml"
        head, rest = pyproject.read_text().split("[project]", 1)
        section, sep, tail = rest.partition("\n[")
        section = re.sub(
            r'^(version\s*=\s*)"[^"]+"', rf'\g<1>"{new}"', section, count=1, flags=re.MULTILINE
        )
        pyproject.write_text(head + "[project]" + section + sep + tail)
        lock = proj / "uv.lock"
        if lock.exists():
            text = re.sub(
                r'^(\[\[package\]\]\nname = "([^"]+)"\nversion = )"[^"]+"',
                lambda m: f'{m.group(1)}"{new}"' if m.group(2) in names else m.group(0),
                lock.read_text(),
                flags=re.MULTILINE,
            )
            lock.write_text(text)
    for init in INIT_FILES:
        init.write_text(
            re.sub(
                r'^__version__ = "[^"]+"',
                f'__version__ = "{new}"',
                init.read_text(),
                flags=re.MULTILINE,
            )
        )
    if CITATION.exists():
        text = re.sub(
            r"^version: .*$", f'version: "{new}"', CITATION.read_text(), flags=re.MULTILINE
        )
        today = datetime.datetime.now(datetime.UTC).date().isoformat()
        text = re.sub(r"^date-released: .*$", f'date-released: "{today}"', text, flags=re.MULTILINE)
        CITATION.write_text(text)
    print(f"version set to {new}; add a CHANGELOG.md entry, then commit and tag v{new}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    check_p = sub.add_parser("check")
    check_p.add_argument("--tag", help="release tag that must match, e.g. v0.9.0")
    set_p = sub.add_parser("set")
    set_p.add_argument("version")
    args = parser.parse_args()
    if args.cmd == "check":
        return check(args.tag)
    if args.cmd == "set":
        set_version(args.version)
        return 0
    print(hub_version())
    return 0


if __name__ == "__main__":
    sys.exit(main())
