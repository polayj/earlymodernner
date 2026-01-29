"""
EarlyModernNER - Named Entity Recognition for Early Modern Documents

Usage:
    python -m earlymodernner --input /path/to/docs --output results.jsonl
    python -m earlymodernner --input document.txt --output results.jsonl
    python -m earlymodernner --input docs/ --output results.csv --csv
    python -m earlymodernner --download  # Pre-download adapters for offline use
"""

import argparse
import sys
from pathlib import Path

from .version import VERSION
from .pipeline import run_pipeline, download_all_adapters


def parse_args():
    parser = argparse.ArgumentParser(
        prog="earlymodernner",
        description="Named Entity Recognition for Early Modern English documents (1500-1800)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single file
    python -m earlymodernner --input document.txt --output results.jsonl

    # Process all files in a directory
    python -m earlymodernner --input /path/to/docs --output results.jsonl

    # Output as CSV instead of JSONL
    python -m earlymodernner --input /path/to/docs --output results.csv --csv

    # Process only specific entity types
    python -m earlymodernner --input docs/ --output results.jsonl --entity-types TOPONYM COMMODITY

    # Pre-download adapters for offline use
    python -m earlymodernner --download

Supported input formats:
    .txt, .md, .xml, .jsonl

Model adapters are automatically downloaded from Hugging Face Hub on first use
and cached in ~/.cache/earlymodernner/
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file or directory containing documents to process",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (.jsonl or .csv)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model adapters from Hugging Face Hub and exit",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV instead of JSONL",
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        nargs="+",
        default=["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"],
        help="Entity types to extract (default: all four)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing LoRA adapters (default: download from Hugging Face Hub)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Documents to process at once (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --download option
    if args.download:
        try:
            download_all_adapters(verbose=True)
            print("\nAdapters downloaded successfully!")
            sys.exit(0)
        except Exception as e:
            print(f"Error downloading adapters: {e}")
            sys.exit(1)

    # Require --input and --output for normal operation
    if not args.input or not args.output:
        print("Error: --input and --output are required (or use --download)")
        sys.exit(1)

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Validate output path
    output_path = Path(args.output)
    if args.csv and not output_path.suffix == ".csv":
        print("Warning: --csv flag set but output file doesn't have .csv extension")

    # Run the pipeline
    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            entity_types=args.entity_types,
            model_dir=args.model_dir,
            output_csv=args.csv,
            batch_size=args.batch_size,
            device=args.device,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
