#!/usr/bin/env python3
"""Build the static, multi-source snapshot used by the codebase city."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import tomllib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = SITE_ROOT / "public" / "city-data.json"
REPOSITORY = "SauersML/gam"

SOURCE_SUFFIXES = {
    ".rs",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".md",
    ".sh",
    ".ts",
    ".tsx",
    ".js",
    ".json",
}
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".buildd",
    ".cargo",
    ".claude",
    ".codex-artifacts",
    ".codex-issue-workers",
    ".config",
    ".next",
    ".wrangler",
    "node_modules",
    "target",
    "dist",
    "__pycache__",
}
ISSUE_REF = re.compile(r"(?<![\w/])#(\d{2,5})\b")
FILENAME_ISSUE = re.compile(r"(?:^|_)(\d{3,5})(?:_|\.|$)")
TEST_MARKER = re.compile(r"#\[(?:tokio::)?test\]|\bdef test_|\bclass Test")
RUST_FAILURE = re.compile(
    r"^- \*\*(FAIL|TIMEOUT|TERMINATING|LEAK)\*\* `([^`]+)` :: `([^`]+)`",
    re.MULTILINE,
)


def run(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def gh_json(endpoint: str, *, paginate: bool = False) -> Any:
    args = ["gh", "api"]
    if paginate:
        args.extend(["--paginate", "--slurp"])
    args.append(endpoint)
    return json.loads(run(args))


def iso_age_days(value: str | None, now: datetime) -> float:
    if not value:
        return 0.0
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return max(0.0, (now - parsed).total_seconds() / 86400)


def raw_district(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "crates":
        return parts[1]
    if parts and parts[0] == "gamfit":
        return "python-api"
    if parts and parts[0] == "tests":
        if len(parts) >= 2 and parts[1] not in {"data", "fixtures"}:
            return f"tests:{parts[1].removesuffix('.rs')}"
        return "integration-tests"
    if parts and parts[0] == "src":
        return "root-engine"
    if parts[:2] == (".github", "workflows"):
        return "actions-control"
    if parts and parts[0] == "bench":
        return "measurement-labs"
    if parts and parts[0] == "docs":
        return "documentation"
    if parts and parts[0] == "examples":
        return "examples"
    return "civic-infrastructure"


def file_kind(path: str) -> str:
    suffix = Path(path).suffix
    if path.startswith("tests/") or "/tests/" in path or Path(path).name.startswith("test_"):
        return "test"
    if path.startswith(".github/workflows/"):
        return "workflow"
    if path.startswith("bench/"):
        return "measurement"
    if suffix == ".md":
        return "documentation"
    if suffix == ".py":
        return "python"
    if suffix == ".rs":
        return "rust"
    return "infrastructure"


def collect_files() -> tuple[list[dict[str, Any]], dict[str, str]]:
    files: list[dict[str, Any]] = []
    file_text: dict[str, str] = {}

    for root, dirs, names in os.walk(REPO_ROOT):
        root_path = Path(root)
        dirs[:] = [
            name
            for name in dirs
            if name not in EXCLUDED_DIRS
            and (root_path / name) != SITE_ROOT
        ]
        for name in names:
            path = root_path / name
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                relative = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            if relative.startswith("tools/codebase-city/"):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > 2_000_000:
                continue
            text = raw.decode("utf-8", errors="replace")
            loc = max(1, text.count("\n") + 1)
            refs = {int(match) for match in ISSUE_REF.findall(text)}
            refs.update(
                int(match)
                for match in FILENAME_ISSUE.findall(path.name)
                if 0 < int(match) < 100_000
            )
            file_text[relative] = text
            files.append(
                {
                    "id": relative,
                    "path": relative,
                    "rawDistrict": raw_district(relative),
                    "kind": file_kind(relative),
                    "loc": loc,
                    "bytes": len(raw),
                    "tests": len(TEST_MARKER.findall(text)),
                    "issueRefs": sorted(refs)[:24],
                }
            )

    mass = Counter()
    for item in files:
        mass[item["rawDistrict"]] += item["loc"]
    keep = {name for name, _ in mass.most_common(18)}
    for item in files:
        district = item.pop("rawDistrict")
        item["district"] = district if district in keep else "outer-boroughs"
    return files, file_text


def collect_issues(now: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pages = gh_json(
        f"repos/{REPOSITORY}/issues?state=all&per_page=100&sort=updated&direction=desc",
        paginate=True,
    )
    raw_issues = [
        item
        for page in pages
        for item in page
        if "pull_request" not in item
    ]
    open_issues = [item for item in raw_issues if item["state"] == "open"]
    recent_closed = [item for item in raw_issues if item["state"] == "closed"][:350]
    represented = open_issues + recent_closed
    issue_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    valid_numbers = {item["number"] for item in raw_issues}
    for item in represented:
        body = item.get("body") or ""
        refs = {
            int(match)
            for match in ISSUE_REF.findall(body)
            if int(match) in valid_numbers and int(match) != item["number"]
        }
        issue_rows.append(
            {
                "number": item["number"],
                "title": item["title"],
                "state": item["state"],
                "comments": item["comments"],
                "createdAgeDays": round(iso_age_days(item["created_at"], now), 2),
                "updatedAgeDays": round(iso_age_days(item["updated_at"], now), 2),
                "closedAgeDays": round(iso_age_days(item.get("closed_at"), now), 2)
                if item.get("closed_at")
                else None,
                "labels": [label["name"] for label in item.get("labels", [])],
                "url": item["html_url"],
                "refs": sorted(refs)[:30],
            }
        )
        edges.extend(
            {"source": item["number"], "target": target, "kind": "issue-reference"}
            for target in refs
        )
    counts = {
        "total": len(raw_issues),
        "open": len(open_issues),
        "closed": len(raw_issues) - len(open_issues),
        "comments": sum(item["comments"] for item in raw_issues),
    }
    return issue_rows, edges, counts


def parse_git_log() -> list[dict[str, Any]]:
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    text = run(
        [
            "git",
            "log",
            "origin/main",
            "-n",
            "500",
            "--name-only",
            "--format=%x1e%H%x1f%aI%x1f%an%x1f%s",
        ]
    )
    commits: list[dict[str, Any]] = []
    for record in text.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        lines = record.splitlines()
        header = lines[0].split("\x1f")
        if len(header) != 4:
            continue
        sha, date, _author, subject = header
        files = [line.strip() for line in lines[1:] if line.strip()]
        refs = sorted({int(match) for match in ISSUE_REF.findall(subject)})
        commits.append(
            {
                "sha": sha[:10],
                "fullSha": sha,
                "date": date,
                "subject": subject,
                "issueRefs": refs,
                "files": files[:80],
                "fileCount": len(files),
                "url": f"https://github.com/{REPOSITORY}/commit/{sha}",
            }
        )
    return commits


def best_file_match(label: str, files: list[dict[str, Any]]) -> str | None:
    tokens = {
        token
        for token in re.split(r"[^a-zA-Z0-9]+", label.lower())
        if len(token) >= 4 and token not in {"test", "tests", "quality", "misc"}
    }
    best: tuple[int, str] | None = None
    for item in files:
        path = item["path"].lower()
        score = sum(1 for token in tokens if token in path)
        if score and (best is None or score > best[0]):
            best = (score, item["id"])
    return best[1] if best else None


def collect_failures(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    master = REPO_ROOT / "bench/gha_results/rust-test-suite/MASTER_FAILURES.md"
    if master.exists():
        text = master.read_text(errors="replace")
        for index, match in enumerate(RUST_FAILURE.finditer(text)):
            status, binary, test = match.groups()
            label = f"{binary} {test}"
            failures.append(
                {
                    "id": f"rust-{index}",
                    "surface": "rust",
                    "status": status,
                    "test": test,
                    "binary": binary,
                    "file": best_file_match(label, files),
                }
            )

    quality = REPO_ROOT / "bench/gha_results/reference-quality/quality_results.tsv"
    if quality.exists():
        with quality.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                outcome = row.get("outcome", "")
                if outcome in {"PASS", "REF_ERROR"}:
                    continue
                test = row.get("test", "")
                failures.append(
                    {
                        "id": f"quality-{row.get('idx', len(failures))}",
                        "surface": "quality",
                        "status": outcome,
                        "test": test,
                        "binary": "quality",
                        "file": best_file_match(test, files),
                    }
                )
    return failures


def collect_runs(now: datetime) -> list[dict[str, Any]]:
    data = gh_json(f"repos/{REPOSITORY}/actions/runs?per_page=100")
    rows: list[dict[str, Any]] = []
    for item in data.get("workflow_runs", []):
        started = item.get("run_started_at") or item.get("created_at")
        updated = item.get("updated_at")
        duration = None
        if started and updated:
            a = datetime.fromisoformat(started.replace("Z", "+00:00"))
            b = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            duration = round(max(0, (b - a).total_seconds() / 60), 1)
        rows.append(
            {
                "id": item["id"],
                "name": item.get("name") or item.get("display_title"),
                "displayTitle": item.get("display_title"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
                "event": item.get("event"),
                "sha": (item.get("head_sha") or "")[:10],
                "ageHours": round(iso_age_days(updated, now) * 24, 2),
                "durationMinutes": duration,
                "url": item.get("html_url"),
            }
        )
    return rows


def collect_measurements(now: datetime) -> list[dict[str, Any]]:
    root = REPO_ROOT / "bench/gha_results"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.glob("*/_run.json"):
        try:
            item = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        completed = item.get("completed_utc")
        rows.append(
            {
                "name": path.parent.name,
                "workflow": item.get("workflow"),
                "status": item.get("job_status"),
                "runId": item.get("run_id"),
                "sha": (item.get("sha") or "")[:10],
                "lagHours": round(iso_age_days(completed, now) * 24, 2),
                "url": item.get("run_url"),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def collect_dependencies(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    package_to_district: dict[str, str] = {}
    cargo_files = [REPO_ROOT / "Cargo.toml", *(REPO_ROOT / "crates").glob("*/Cargo.toml")]
    parsed: list[tuple[str, dict[str, Any]]] = []
    for path in cargo_files:
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        package = data.get("package", {}).get("name")
        if not package:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        district = raw_district(relative)
        package_to_district[package] = district
        parsed.append((package, data))

    active_districts = {item["district"] for item in files}
    edges: Counter[tuple[str, str]] = Counter()
    for package, data in parsed:
        source = package_to_district.get(package, "civic-infrastructure")
        if source not in active_districts:
            source = "outer-boroughs"
        sections = [
            data.get("dependencies", {}),
            data.get("dev-dependencies", {}),
            data.get("build-dependencies", {}),
        ]
        for dependencies in sections:
            for dependency in dependencies:
                if dependency not in package_to_district:
                    continue
                target = package_to_district[dependency]
                if target not in active_districts:
                    target = "outer-boroughs"
                if source != target:
                    edges[(source, target)] += 1
    return [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in edges.most_common()
    ]


def collect_history(files: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    current = {
        item["id"]: item["bytes"]
        for item in files
    }
    snapshots: list[dict[str, Any]] = [
        {
            "ageDays": 0,
            "label": "worktree now",
            "sha": "worktree",
            "date": now.isoformat(),
            "files": [[path, size] for path, size in current.items()],
            "fileCount": len(current),
            "bytes": sum(current.values()),
        }
    ]
    seen_shas: set[str] = set()
    for days in (1, 3, 7, 14, 30, 60, 90, 180, 365):
        cutoff = datetime.fromtimestamp(now.timestamp() - days * 86400, UTC).isoformat()
        try:
            sha = run(
                ["git", "rev-list", "-1", f"--before={cutoff}", "origin/main"]
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if not sha or sha in seen_shas:
            continue
        seen_shas.add(sha)
        try:
            tree = run(["git", "ls-tree", "-r", "--long", sha])
            date = run(["git", "show", "-s", "--format=%aI", sha]).strip()
        except subprocess.CalledProcessError:
            continue
        file_sizes: list[list[Any]] = []
        total_bytes = 0
        for line in tree.splitlines():
            metadata, separator, path = line.partition("\t")
            if not separator:
                continue
            parts = metadata.split()
            if len(parts) < 4 or not parts[3].isdigit():
                continue
            if Path(path).suffix.lower() not in SOURCE_SUFFIXES:
                continue
            size = int(parts[3])
            file_sizes.append([path, size])
            total_bytes += size
        snapshots.append(
            {
                "ageDays": days,
                "label": f"{days}d ago",
                "sha": sha[:10],
                "date": date,
                "files": file_sizes,
                "fileCount": len(file_sizes),
                "bytes": total_bytes,
            }
        )
    return snapshots


def build() -> None:
    now = datetime.now(UTC)
    repo = gh_json(f"repos/{REPOSITORY}")
    files, _ = collect_files()
    issues, issue_edges, issue_counts = collect_issues(now)
    commits = parse_git_log()
    failures = collect_failures(files)
    runs = collect_runs(now)
    measurements = collect_measurements(now)
    dependencies = collect_dependencies(files)
    history = collect_history(files, now)

    commit_file_edges: list[dict[str, Any]] = []
    commit_issue_edges: list[dict[str, Any]] = []
    file_ids = {item["id"] for item in files}
    represented_issues = {item["number"] for item in issues}
    for commit in commits[:220]:
        commit_file_edges.extend(
            {"commit": commit["sha"], "file": path}
            for path in commit["files"][:18]
            if path in file_ids
        )
        commit_issue_edges.extend(
            {"commit": commit["sha"], "issue": issue}
            for issue in commit["issueRefs"]
            if issue in represented_issues
        )

    district_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "loc": 0, "tests": 0, "failures": 0}
    )
    file_to_district = {item["id"]: item["district"] for item in files}
    for item in files:
        district_counts[item["district"]]["files"] += 1
        district_counts[item["district"]]["loc"] += item["loc"]
        district_counts[item["district"]]["tests"] += item["tests"]
    for failure in failures:
        district = file_to_district.get(failure.get("file") or "")
        if district:
            district_counts[district]["failures"] += 1
    districts = [
        {"id": district, **counts}
        for district, counts in sorted(
            district_counts.items(), key=lambda pair: pair[1]["loc"], reverse=True
        )
    ]

    recent_commits = sum(
        1
        for commit in commits
        if iso_age_days(commit["date"], now) <= 7
    )
    summary = {
        "files": len(files),
        "loc": sum(item["loc"] for item in files),
        "tests": sum(item["tests"] for item in files),
        "failures": len(failures),
        "openIssues": issue_counts["open"],
        "totalIssues": issue_counts["total"],
        "comments": issue_counts["comments"],
        "commits7d": recent_commits,
        "runs": len(runs),
        "stars": repo.get("stargazers_count", 0),
        "measurementLagHours": round(
            max((item["lagHours"] for item in measurements), default=0), 1
        ),
    }

    payload = {
        "generatedAt": now.isoformat(),
        "repository": {
            "name": REPOSITORY,
            "url": repo.get("html_url"),
            "defaultBranch": repo.get("default_branch"),
            "description": repo.get("description"),
        },
        "summary": summary,
        "districts": districts,
        "files": files,
        "issues": issues,
        "issueEdges": issue_edges,
        "issueCounts": issue_counts,
        "commits": commits,
        "commitFileEdges": commit_file_edges,
        "commitIssueEdges": commit_issue_edges,
        "failures": failures,
        "runs": runs,
        "measurements": measurements,
        "dependencies": dependencies,
        "history": history,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"wrote {OUTPUT.relative_to(SITE_ROOT)} "
        f"({OUTPUT.stat().st_size / 1_000_000:.2f} MB, "
        f"{len(files)} buildings, {len(issues)} issues, "
        f"{len(commits)} commits, {len(failures)} failures)"
    )


if __name__ == "__main__":
    build()
