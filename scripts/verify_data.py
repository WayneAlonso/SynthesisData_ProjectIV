"""Verify shipped data bytes, table dimensions, and configured dataset paths."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            errors.append(f"Missing: {item['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            errors.append(f"Content changed: {item['path']}")
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                columns = len(next(reader, []))
                rows = 0
                malformed = 0
                for row in reader:
                    rows += 1
                    malformed += len(row) != columns
            if columns != item["columns"] or rows != item["rows"] or malformed:
                errors.append(f"Table shape mismatch: {item['path']}")
            print(f"{path.name}: {rows:,} rows x {columns} columns")
    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    shipped = {item["path"] for item in manifest["files"]}
    for name, dataset in config["dataset_catalog"].items():
        if dataset["path"] not in shipped or not (ROOT / dataset["path"]).is_file():
            errors.append(f"Configured dataset is not shipped: {name}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"PASS: {len(manifest['files'])} data files verified; all configured datasets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
