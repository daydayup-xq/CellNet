"""Validate an installed Image Atlas against the CellNet fitted assets."""

from pathlib import Path
import argparse
import json
from cellnet.pretraining_data import verify_image_atlas


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=root,
                        help="CellNet clone containing assets/ and weights/")
    parser.add_argument("--data", type=Path,
                        help="Defaults to datasets/pretraining/full/ under --release")
    parser.add_argument("--samples", type=int, default=2,
                        help="Number of native records to inspect (default: 2)")
    parser.add_argument("--checksums", action="store_true",
                        help="Read all selected files and compare available trusted checksums")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--storage-classes", nargs="+")
    selection.add_argument("--shards", nargs="+")
    parser.add_argument("--output", type=Path, help="Optional JSON report; existing files are preserved")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    data = args.data if args.data is not None else args.release / "datasets/pretraining/full"
    report = verify_image_atlas(
        data, release=args.release, samples=args.samples, checksums=args.checksums,
        storage_classes=args.storage_classes, shards=args.shards,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    summary = {key: value for key, value in report.items() if key != "computed_sha256"}
    if "computed_sha256" in report:
        summary["computed_sha256_entries"] = len(report["computed_sha256"])
        if args.output is not None:
            summary["checksum_report"] = str(args.output)
        else:
            summary["checksum_report"] = "Use --output to save individual file digests"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
