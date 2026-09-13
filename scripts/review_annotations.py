"""Prepare a review packet, or export a fully human-reviewed packet."""
import argparse
import json
from pathlib import Path

from src.annotation_review import export_review, prepare_review
from src.config import CHROMA_PATH, QUESTIONS_PATH
from src.provenance import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "export"])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--catalog", type=Path, default=CHROMA_PATH / "chunks.json")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.packet.exists():
            parser.error("Packet already exists; refusing to overwrite human review")
        args.packet.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.packet, prepare_review(args.questions, args.catalog))
    else:
        if args.output is None or args.output.exists():
            parser.error("Choose a new --output path")
        labels = export_review(json.loads(args.packet.read_text()), args.questions, args.catalog)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, labels)


if __name__ == "__main__":
    main()
