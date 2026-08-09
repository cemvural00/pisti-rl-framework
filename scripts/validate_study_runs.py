"""Validate all confirmatory run artifacts before scientific evaluation."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import yaml
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]


def expected_runs() -> Iterable[Tuple[str, int, bool, str]]:
    for seed in range(10):
        yield f"study_mem_on_s{seed}", seed, True, "league"
        yield f"study_mem_off_s{seed}", seed, False, "league"
    for seed in range(5):
        yield f"study_fixed_s{seed}", seed, True, "fixed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_one(name: str, seed: int, memory: bool, regime: str) -> Dict:
    run_dir = ROOT / "runs" / name
    required = ["config.yaml", "metadata.json", "eval.csv", "final_model.zip"]
    missing = [item for item in required if not (run_dir / item).is_file()]
    if missing:
        raise ValueError(f"{name}: missing {missing}")

    with (run_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    with (run_dir / "eval.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    checks = {
        "run_name": config["run_name"] == name,
        "seed": config["seed"] == seed,
        "memory": config["observer"]["memory"] is memory,
        "steps": config["total_timesteps"] == 6_000_000,
        "metadata_config": metadata["config"] == config,
        "finished": bool(metadata.get("finished")),
        "final_evaluation": int(rows[-1]["timesteps"]) == 6_000_000,
        "evaluation_rows": len(rows) == 25,
    }
    curriculum_keys = set(config["curriculum"][-1]["weights"])
    checks["regime"] = (
        {"pool", "latest"} <= curriculum_keys
        if regime == "league"
        else curriculum_keys.isdisjoint({"pool", "latest"})
    )

    model = MaskablePPO.load(run_dir / "final_model.zip", device="cpu")
    checks["finite_parameters"] = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.policy.parameters()
    )
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"{name}: failed {failed}")

    return {
        "seed": seed,
        "memory": memory,
        "regime": regime,
        "wall_seconds": metadata["wall_seconds"],
        "git_hash": metadata["git_hash"],
        "git_dirty": metadata["git_dirty"],
        "python": metadata["python"],
        "packages": metadata["packages"],
        "model_bytes": (run_dir / "final_model.zip").stat().st_size,
        "model_sha256": sha256(run_dir / "final_model.zip"),
        "checks": checks,
    }


def main() -> None:
    runs = {
        name: validate_one(name, seed, memory, regime)
        for name, seed, memory, regime in expected_runs()
    }
    git_hashes = {run["git_hash"] for run in runs.values()}
    python_versions = {run["python"] for run in runs.values()}
    package_sets = {json.dumps(run["packages"], sort_keys=True) for run in runs.values()}
    if len(git_hashes) != 1 or len(python_versions) != 1 or len(package_sets) != 1:
        raise ValueError("study runs do not share one recorded software environment")
    report = {
        "status": "pass",
        "n_runs": len(runs),
        "shared_environment": {
            "git_hash": next(iter(git_hashes)),
            "git_dirty": all(run["git_dirty"] for run in runs.values()),
            "python": next(iter(python_versions)),
            "packages": json.loads(next(iter(package_sets))),
        },
        "runs": runs,
    }
    output = ROOT / "results/study_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"validated {len(runs)} runs -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
