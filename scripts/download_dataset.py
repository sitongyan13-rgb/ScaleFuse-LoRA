#!/usr/bin/env python
"""Resumable downloader for the officially designated ODinW-13 archives.

The GLIP project designates GLIPModel/GLIP on Hugging Face as the ODinW
source. This script corrects the upstream helper's ``tree`` URL to the
downloadable ``resolve`` URL, preserves archives, validates ZIP structure,
extracts safely, and records actual hashes after transfer.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


ODINW13 = (
    "AerialMaritimeDrone",
    "Aquarium",
    "CottontailRabbits",
    "EgoHands",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "pistols",
    "pothole",
    "Raccoon",
    "ShellfishOpenImages",
    "thermalDogsAndPeople",
    "VehiclesOpenImages",
)
BASE_URL = "https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35"
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(str(archive)) as zf:
        corrupt = zf.testzip()
        if corrupt:
            raise RuntimeError("{} contains corrupt member {!r}".format(archive, corrupt))
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                raise RuntimeError(
                    "Refusing path traversal member {!r} in {}".format(
                        info.filename, archive
                    )
                )
        zf.extractall(str(destination))


def remote_size(session: requests.Session, url: str, timeout: float) -> Optional[int]:
    response = session.head(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def download_one(
    session: requests.Session,
    url: str,
    target: Path,
    timeout: float,
    retries: int,
) -> Dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(session, url, timeout)
    current = target.stat().st_size if target.exists() else 0
    if expected is not None and current > expected:
        raise RuntimeError(
            "{} is larger than remote object ({} > {}); move it aside manually".format(
                target, current, expected
            )
        )
    if expected is not None and current == expected:
        print("[complete] {} ({} bytes)".format(target, expected), flush=True)
    else:
        for attempt in range(1, retries + 1):
            current = target.stat().st_size if target.exists() else 0
            headers = {"Range": "bytes={}-".format(current)} if current else {}
            try:
                with session.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(timeout, timeout),
                ) as response:
                    response.raise_for_status()
                    append = current > 0 and response.status_code == 206
                    if current > 0 and not append:
                        raise RuntimeError(
                            "Server ignored Range for existing partial file {}; "
                            "move the partial file aside before retrying".format(target)
                        )
                    mode = "ab" if append else "wb"
                    with target.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                handle.write(chunk)
                                handle.flush()
                    break
            except Exception:
                if attempt == retries:
                    raise
                delay = min(30, 2 ** attempt)
                print(
                    "[retry {}/{}] {} after {} seconds".format(
                        attempt, retries, target.name, delay
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        actual = target.stat().st_size
        if expected is not None and actual != expected:
            raise RuntimeError(
                "Size mismatch for {}: actual {}, expected {}".format(
                    target, actual, expected
                )
            )
    return {
        "url": url,
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def parse_selection(value: str) -> List[str]:
    if value.lower() in ("all", "odinw13"):
        return list(ODINW13)
    selected = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(selected).difference(ODINW13))
    if unknown:
        raise ValueError(
            "Unknown ODinW-13 member(s): {}. Valid values: {}".format(
                ", ".join(unknown), ", ".join(ODINW13)
            )
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members",
        default="all",
        help="'all' or a comma-separated subset of the 13 official archive names",
    )
    parser.add_argument("--archive-dir", default="data/archives/odinw13")
    parser.add_argument("--extract-dir", default="data/odinw")
    parser.add_argument(
        "--manifest", default="data/odinw13_download_manifest.json"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=5.0,
        help="Refuse download when archive drive has less free space",
    )
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--offline-existing",
        action="store_true",
        help=(
            "Use already-downloaded <archive-dir>/<member>.zip files without "
            "any network request; verify, hash, and optionally extract them"
        ),
    )
    parser.add_argument(
        "--use-environment-proxy",
        action="store_true",
        help="By default requests ignores proxy environment variables because this host's configured proxy failed audit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    members = parse_selection(args.members)
    archive_dir = Path(args.archive_dir)
    extract_dir = Path(args.extract_dir)
    manifest_path = Path(args.manifest)
    archive_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(archive_dir.resolve())).free
    required_free = int(args.min_free_gib * (1024 ** 3))
    print("archive_dir={}".format(archive_dir.resolve()))
    print("extract_dir={}".format(extract_dir.resolve()))
    print("free_bytes={}".format(free))
    print("selected={}".format(",".join(members)))
    if free < required_free:
        raise RuntimeError(
            "Only {} free bytes; minimum is {}".format(free, required_free)
        )
    if args.dry_run:
        for member in members:
            print("{}/{}.zip".format(BASE_URL, member))
        return 0

    session = requests.Session()
    session.trust_env = args.use_environment_proxy
    session.headers.update({"User-Agent": "groundingdino-research-audit/1.0"})
    records = []
    for member in members:
        url = "{}/{}.zip?download=true".format(BASE_URL, member)
        target = archive_dir / "{}.zip".format(member)
        if args.offline_existing:
            if not target.is_file():
                raise FileNotFoundError(
                    "Offline archive is missing: {}".format(target)
                )
            with zipfile.ZipFile(str(target)) as zf:
                corrupt = zf.testzip()
            if corrupt:
                raise RuntimeError(
                    "{} contains corrupt member {!r}".format(target, corrupt)
                )
            print("[offline/verified] {}".format(target), flush=True)
            record = {
                "url": url,
                "path": str(target),
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "zip_test": "ok",
                "transfer": "provided_local_archive",
            }
        else:
            print("[download] {} <- {}".format(target, url), flush=True)
            record = download_one(
                session, url, target, args.timeout, args.retries
            )
            record["transfer"] = "downloaded_or_resumed"
        record["dataset"] = member
        if not args.no_extract:
            print("[verify/extract] {}".format(member), flush=True)
            safe_extract(target, extract_dir)
            record["extracted_to"] = str(extract_dir)
        records.append(record)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": "ODinW-13",
                    "official_designated_source": BASE_URL,
                    "archive_mode": (
                        "offline_existing"
                        if args.offline_existing
                        else "download_or_resume"
                    ),
                    "members": records,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    print("manifest={}".format(manifest_path.resolve()))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        raise
