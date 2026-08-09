"""Create a checksum manifest for confirmatory source, models, and results."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> Iterable[Path]:
    for directory in ("agents", "analysis", "encoding", "engine", "envs", "training"):
        yield from sorted((ROOT / directory).glob("*.py"))
    yield from sorted((ROOT / "configs").glob("*.yaml"))
    yield from sorted((ROOT / "scripts").glob("*.py"))
    yield from sorted((ROOT / "scripts").glob("*.sh"))
    yield from sorted((ROOT / "tests").glob("*.py"))
    for name in ("pyproject.toml", "requirements-research.txt"):
        yield ROOT / name
    yield from sorted((ROOT / "research").glob("*.md"))
    yield from sorted((ROOT / "paper").glob("*.tex"))
    yield ROOT / "paper/references.bib"
    yield ROOT / "paper/main.pdf"
    yield ROOT / "BLOG_OUTLINE.md"
    yield from sorted((ROOT / "runs").glob("study_*/final_model.zip"))
    yield from sorted((ROOT / "runs").glob("study_*/config.yaml"))
    yield from sorted((ROOT / "runs").glob("study_*/metadata.json"))
    yield from sorted((ROOT / "runs/robustness_attackers").glob("*.zip"))
    yield from sorted((ROOT / "results").glob("memory_study*.json"))
    yield from sorted((ROOT / "results/robustness").glob("*.json"))
    yield ROOT / "results/robustness_study_summary.json"
    yield ROOT / "results/study_validation.json"
    yield ROOT / "results/study_training_summary.json"
    yield ROOT / "plots/memory_study_effects.png"
    yield ROOT / "plots/robustness_study.png"
    yield ROOT / "plots/study_training_curves.png"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build() -> Dict:
    files = {}
    for path in selected_files():
        if path.exists():
            files[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "protocol": "research/experiment_preregistration.md",
        "files": files,
    }


def main() -> None:
    output = ROOT / "research/artifact_manifest.json"
    output.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
